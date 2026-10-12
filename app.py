"""Voice Chain dashboard: mic picker, live meters, sliders, presets, tray icon + mini widget."""
import queue, sys, threading
import customtkinter as ctk
import pystray
from PIL import Image, ImageDraw
import engine

GREEN, GREEN_DK, RED, BG, CARD, TXT, DIM = "#76B900", "#4d7a00", "#e5484d", "#101010", "#1b1b1b", "#e8e8e8", "#8a8a8a"

# (key, label, min, max, step, fmt)  -- every slider maps 1:1 to a Chain config key
SECTIONS = [
    ("NOISE REMOVAL", "gate_on", [
        ("exp_strength", "Strength", 0.0, 3.0, 0.05, "{:.2f}x"),
        ("floor_db", "Gate floor", -60, -10, 1, "{:.0f} dB"),
        ("vad_open", "Voice sensitivity", 0.2, 0.95, 0.01, "{:.2f}"),
        ("hold_ms", "Gate hold", 0, 400, 5, "{:.0f} ms"),
        ("release_ms", "Gate release", 40, 600, 5, "{:.0f} ms"),
        ("passes", "Denoise passes", 1, 3, 1, "{:.0f}"),
        ("dfn_atten", "DeepFilter depth", 0, 100, 1, "{:.0f} dB"),
        ("dfn_pf", "DeepFilter post-filter", 0.0, 0.05, 0.001, "{:.3f}"),
    ]),
    ("TONE / EQ", "eq_on", [
        ("hpf_hz", "Low cut", 20, 200, 1, "{:.0f} Hz"),
        ("lowmid_db", "Low-mid (muddy)", -6, 3, 0.1, "{:+.1f} dB"),
        ("lowmid_hz", "Low-mid freq", 150, 600, 5, "{:.0f} Hz"),
        ("presence_db", "Presence", -3, 6, 0.1, "{:+.1f} dB"),
        ("presence_hz", "Presence freq", 2000, 6000, 50, "{:.0f} Hz"),
        ("air_db", "Air", -3, 6, 0.1, "{:+.1f} dB"),
    ]),
    ("DYNAMICS", "comp_on", [
        ("comp_thresh_db", "Compressor threshold", -40, 0, 0.5, "{:.1f} dB"),
        ("comp_ratio", "Compressor ratio", 1, 8, 0.1, "{:.1f}:1"),
        ("makeup_db", "Makeup gain", -6, 12, 0.5, "{:+.1f} dB"),
        ("limiter_db", "Limiter ceiling", -6, 0, 0.5, "{:.1f} dB"),
    ]),
]
ENGINES = {"Fast (RNNoise)": "rnnoise", "Quality (DeepFilterNet3)": "dfn"}
ENGINE_LABEL = {v: k for k, v in ENGINES.items()}
SECTION_SWITCH_LABEL = {"gate_on": "VAD gate", "eq_on": "EQ", "comp_on": "Compressor"}


def tray_image(on=True):
    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((4, 4, 60, 60), fill=GREEN if on else "#555555")
    for i, h in enumerate((14, 26, 36, 26, 14)):
        x = 14 + i * 9
        d.rounded_rectangle((x, 32 - h // 2, x + 5, 32 + h // 2), radius=2, fill="#101010")
    return im


def meter_val(db):
    return max(0.0, min(1.0, (float(db) + 70.0) / 70.0))


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        self.title("Voice Chain")
        self.geometry("1040x760")
        self.minsize(900, 640)
        self.configure(fg_color=BG)
        self.eng = engine.Engine()
        self.presets = engine.load_presets()
        self.settings = engine.load_settings()
        self.cmds = queue.Queue()  # tray thread -> tk thread
        self.sliders, self.labels, self.switches = {}, {}, {}
        self._save_job = None
        self._cache = {}
        self.mini = None

        if self.settings.get("cfg"):
            self.eng.set(**{k: v for k, v in self.settings["cfg"].items() if k in engine.CFG})

        self._build()
        self._build_tray()
        self.protocol("WM_DELETE_WINDOW", self.hide)
        self._populate_devices()
        if self.settings.get("enabled") and self.dev_var.get() in self.devices:
            self.power.select(); self._toggle_power()
        self.after(50, self._tick)

    # ------------------------------------------------------------------ layout
    def _card(self, parent, title, **kw):
        f = ctk.CTkFrame(parent, fg_color=CARD, corner_radius=10, **kw)
        ctk.CTkLabel(f, text=title, text_color=GREEN, font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=14, pady=(10, 4))
        return f

    def _build(self):
        top = ctk.CTkFrame(self, fg_color="#0a0a0a", corner_radius=0, height=64)
        top.pack(fill="x")
        ctk.CTkLabel(top, text="VOICE", text_color=GREEN, font=ctk.CTkFont(size=22, weight="bold")).pack(side="left", padx=(20, 4), pady=14)
        ctk.CTkLabel(top, text="CHAIN", text_color=TXT, font=ctk.CTkFont(size=22, weight="bold")).pack(side="left")
        self.status = ctk.CTkLabel(top, text="Off", text_color=DIM)
        self.status.pack(side="right", padx=20)
        self.power = ctk.CTkSwitch(top, text="ON / OFF", progress_color=GREEN, command=self._toggle_power, font=ctk.CTkFont(weight="bold"))
        self.power.pack(side="right", padx=10)
        self.bypass = ctk.CTkSwitch(top, text="Bypass (A/B)", progress_color="#d9822b", command=self._toggle_bypass)
        self.bypass.pack(side="right", padx=10)

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=14, pady=14)
        body.grid_columnconfigure(1, weight=1); body.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(body, fg_color="transparent", width=300)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 12))

        # microphone
        mic = self._card(left, "MICROPHONE"); mic.pack(fill="x", pady=(0, 10))
        self.dev_var = ctk.StringVar()
        self.dev_menu = ctk.CTkOptionMenu(mic, variable=self.dev_var, values=["(none)"], width=270, fg_color="#262626",
                                          button_color=GREEN_DK, button_hover_color=GREEN, command=self._device_changed)
        self.dev_menu.pack(padx=14, pady=(0, 6))
        row = ctk.CTkFrame(mic, fg_color="transparent"); row.pack(fill="x", padx=14, pady=(0, 6))
        ctk.CTkButton(row, text="Refresh devices", width=130, fg_color="#262626", hover_color="#333", command=self._refresh).pack(side="left")
        self.out_lbl = ctk.CTkLabel(mic, text="", text_color=DIM, wraplength=270, justify="left", font=ctk.CTkFont(size=11))
        self.out_lbl.pack(anchor="w", padx=14, pady=(0, 10))

        # meters
        mt = self._card(left, "LIVE"); mt.pack(fill="x", pady=(0, 10))
        self.bars = {}
        for key, label in (("in", "Input"), ("out", "Output"), ("vad", "Voice detect")):
            r = ctk.CTkFrame(mt, fg_color="transparent"); r.pack(fill="x", padx=14, pady=3)
            ctk.CTkLabel(r, text=label, width=84, anchor="w", text_color=DIM).pack(side="left")
            b = ctk.CTkProgressBar(r, width=130, progress_color=GREEN, fg_color="#2a2a2a"); b.set(0); b.pack(side="left", padx=4)
            v = ctk.CTkLabel(r, text="", width=52, anchor="e", text_color=TXT, font=ctk.CTkFont(size=11)); v.pack(side="left")
            self.bars[key] = (b, v)
        gr = ctk.CTkFrame(mt, fg_color="transparent"); gr.pack(fill="x", padx=14, pady=(3, 12))
        ctk.CTkLabel(gr, text="Gate", width=84, anchor="w", text_color=DIM).pack(side="left")
        self.gate_lbl = ctk.CTkLabel(gr, text="shut", text_color=DIM); self.gate_lbl.pack(side="left", padx=4)
        self.exp_lbl = ctk.CTkLabel(gr, text="", text_color=DIM, font=ctk.CTkFont(size=11)); self.exp_lbl.pack(side="right")
        lr = ctk.CTkFrame(mt, fg_color="transparent"); lr.pack(fill="x", padx=14, pady=(0, 10))
        self.lat_lbl = ctk.CTkLabel(lr, text="", text_color=DIM, font=ctk.CTkFont(size=11)); self.lat_lbl.pack(side="right")

        # master output volume (after the limiter; not part of presets)
        ov = self._card(left, "OUTPUT VOLUME"); ov.pack(fill="x", pady=(0, 10))
        orow = ctk.CTkFrame(ov, fg_color="transparent"); orow.pack(fill="x", padx=14, pady=(0, 12))
        vs = ctk.CTkSlider(orow, from_=-40, to=6, number_of_steps=92, progress_color=GREEN, button_color=GREEN,
                           button_hover_color="#9be000", command=lambda v: self._slide("out_db", v))
        vs.set(self.eng.get()["out_db"]); vs.pack(side="left", fill="x", expand=True)
        vl = ctk.CTkLabel(orow, text="{:+.1f} dB".format(self.eng.get()["out_db"]), width=64, anchor="e", text_color=GREEN)
        vl.pack(side="left", padx=(8, 0))
        self.sliders["out_db"], self.labels["out_db"] = vs, (vl, "{:+.1f} dB")

        # presets
        pr = self._card(left, "PRESETS"); pr.pack(fill="x")
        self.preset_var = ctk.StringVar(value=self.settings.get("preset", "Strong (G)"))
        self.preset_menu = ctk.CTkOptionMenu(pr, variable=self.preset_var, values=list(self.presets), width=270, fg_color="#262626",
                                             button_color=GREEN_DK, button_hover_color=GREEN, command=self._load_preset)
        self.preset_menu.pack(padx=14, pady=(0, 6))
        pb = ctk.CTkFrame(pr, fg_color="transparent"); pb.pack(fill="x", padx=14, pady=(0, 12))
        ctk.CTkButton(pb, text="Save as...", width=90, fg_color=GREEN_DK, hover_color=GREEN, command=self._save_preset).pack(side="left")
        ctk.CTkButton(pb, text="Delete", width=70, fg_color="#262626", hover_color="#5a1f22", command=self._delete_preset).pack(side="left", padx=6)
        ctk.CTkButton(pb, text="Mini widget", width=90, fg_color="#262626", hover_color="#333", command=self.show_mini).pack(side="right")

        # sliders
        right = ctk.CTkScrollableFrame(body, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew")
        cfg = self.eng.get()
        for title, sw_key, rows in SECTIONS:
            card = self._card(right, title); card.pack(fill="x", pady=(0, 10))
            sw = ctk.CTkSwitch(card, text=SECTION_SWITCH_LABEL[sw_key], progress_color=GREEN, command=lambda k=sw_key: self._switch(k))
            sw.place(relx=1.0, x=-14, y=10, anchor="ne")
            (sw.select if cfg[sw_key] else sw.deselect)()
            self.switches[sw_key] = sw
            if sw_key == "gate_on":  # denoiser engine selector lives at the top of the noise card
                er = ctk.CTkFrame(card, fg_color="transparent"); er.pack(fill="x", padx=14, pady=(0, 4))
                ctk.CTkLabel(er, text="Engine", width=150, anchor="w", text_color=TXT).pack(side="left")
                self.engine_btn = ctk.CTkSegmentedButton(er, values=list(ENGINES), selected_color=GREEN_DK,
                                                         selected_hover_color=GREEN, command=self._engine_changed)
                self.engine_btn.set(ENGINE_LABEL[cfg["denoiser"]]); self.engine_btn.pack(side="left", padx=8)
                self.dfn_lbl = ctk.CTkLabel(er, text="", text_color=DIM, font=ctk.CTkFont(size=11)); self.dfn_lbl.pack(side="left", padx=6)
            for key, label, lo, hi, step, fmt in rows:
                r = ctk.CTkFrame(card, fg_color="transparent"); r.pack(fill="x", padx=14, pady=2)
                ctk.CTkLabel(r, text=label, width=150, anchor="w", text_color=TXT).pack(side="left")
                n = max(1, int(round((hi - lo) / step)))
                s = ctk.CTkSlider(r, from_=lo, to=hi, number_of_steps=n, progress_color=GREEN, button_color=GREEN,
                                  button_hover_color="#9be000", command=lambda v, k=key: self._slide(k, v))
                s.set(cfg[key]); s.pack(side="left", fill="x", expand=True, padx=8)
                lb = ctk.CTkLabel(r, text=fmt.format(cfg[key]), width=78, anchor="e", text_color=GREEN)
                lb.pack(side="left")
                self.sliders[key], self.labels[key] = s, (lb, fmt)
            ctk.CTkFrame(card, fg_color="transparent", height=8).pack()

    # ------------------------------------------------------------------ events
    def _slide(self, key, v):
        v = int(round(v)) if key == "passes" else float(v)
        lb, fmt = self.labels[key]; lb.configure(text=fmt.format(v))
        self.eng.set(**{key: v}); self._schedule_save()
        self.mini_sync()

    def _engine_changed(self, label):
        self.eng.set(denoiser=ENGINES[label]); self._schedule_save()

    def _switch(self, key):
        self.eng.set(**{key: bool(self.switches[key].get())}); self._schedule_save()

    def _toggle_power(self):
        if self.power.get():
            if not self.eng.start(self.dev_var.get()):
                self.power.deselect()
        else:
            self.eng.stop()
        self.settings["enabled"] = bool(self.power.get()); self._schedule_save(); self._sync_power_ui()

    def _toggle_bypass(self):
        self.eng.bypass = bool(self.bypass.get())

    def _device_changed(self, _=None):
        self.settings["device"] = self.dev_var.get(); self._schedule_save()
        if self.eng.running:
            if not self.eng.start(self.dev_var.get()):
                self.power.deselect(); self._sync_power_ui()

    def _populate_devices(self):
        self.devices = engine.list_inputs()
        self.dev_menu.configure(values=self.devices or ["(no microphones found)"])
        want = self.settings.get("device")
        self.dev_var.set(want if want in self.devices else (self.devices[0] if self.devices else "(no microphones found)"))
        out = engine.default_output()
        self.out_lbl.configure(text=("Output -> " + out + "\nSelect 'CABLE Output' as the mic in other apps.") if out
                               else "Virtual cable not found - install VB-Cable.", text_color=DIM if out else RED)

    def _refresh(self):
        was = self.eng.running
        self.eng.stop()
        try: engine.refresh_devices()
        except Exception as e: self.status.configure(text=f"refresh failed: {e}", text_color=RED)
        self._populate_devices()
        if was and self.dev_var.get() in self.devices:
            self.eng.start(self.dev_var.get())
        self._sync_power_ui()

    def _load_preset(self, name):
        cfg = {k: v for k, v in self.presets[name].items() if k != "out_db"}  # master volume is not part of a preset
        self.eng.set(**cfg)
        for k, s in self.sliders.items():
            if k not in cfg: continue
            s.set(cfg[k]); lb, fmt = self.labels[k]; lb.configure(text=fmt.format(cfg[k]))
        for k, sw in self.switches.items():
            (sw.select if cfg[k] else sw.deselect)()
        self.engine_btn.set(ENGINE_LABEL[cfg["denoiser"]])
        self.settings["preset"] = name; self._schedule_save(); self.mini_sync()

    def _save_preset(self):
        dlg = ctk.CTkInputDialog(text="Preset name:", title="Save preset")
        name = (dlg.get_input() or "").strip()
        if not name: return
        cfg = {k: v for k, v in self.eng.get().items() if k in engine.CFG and k != "out_db"}
        engine.save_user_preset(name, cfg)
        self.presets = engine.load_presets()
        self.preset_menu.configure(values=list(self.presets)); self.preset_var.set(name)
        self.settings["preset"] = name; self._schedule_save()

    def _delete_preset(self):
        name = self.preset_var.get()
        if name in engine.DEFAULT_PRESETS: return  # built-ins stay
        engine.delete_user_preset(name)
        self.presets = engine.load_presets()
        self.preset_menu.configure(values=list(self.presets)); self.preset_var.set("Strong (G)"); self._load_preset("Strong (G)")

    def _schedule_save(self):
        if self._save_job: self.after_cancel(self._save_job)
        self._save_job = self.after(600, self._save_now)

    def _save_now(self):
        self.settings["cfg"] = {k: v for k, v in self.eng.get().items() if k in engine.CFG}
        engine.save_settings(self.settings)

    def _sync_power_ui(self):
        on = self.eng.running
        (self.power.select if on else self.power.deselect)()
        self.status.configure(text=("Live -> " + (self.eng.output_name or "")) if on else (self.eng.error or "Off"),
                              text_color=GREEN if on else (RED if self.eng.error else DIM))
        if self.tray: self.tray.icon = tray_image(on)
        self.mini_sync()

    # ------------------------------------------------------------------ meters / loop
    def _upd(self, w, **kw):
        """configure() only when something actually changed - every configure on a customtkinter widget redraws it."""
        key = tuple(sorted(kw.items()))
        if self._cache.get(w) != key:
            self._cache[w] = key
            w.configure(**kw)

    def _bar(self, b, val, color=GREEN):
        q = round(max(0.0, min(1.0, float(val))) * 40) / 40  # 2.5% steps: invisible to the eye, 40x fewer redraws
        if self._cache.get(b) != (q, color):
            self._cache[b] = (q, color)
            b.configure(progress_color=color); b.set(q)

    def _tick(self):
        try:
            while True: self.cmds.get_nowait()()
        except queue.Empty:
            pass
        if self.eng.running != bool(self.power.get()):  # engine died / stopped elsewhere
            self._sync_power_ui()
        visible = self.state() == "normal"
        mini_vis = bool(self.mini and self.mini.winfo_exists() and self.mini.state() == "normal")
        if not visible and not mini_vis:  # hidden in the tray: no drawing at all, just poll for tray commands
            self.after(250, self._tick); return
        m = self.eng.meters()
        if visible:
            for key, db in (("in", m["in_db"]), ("out", m["out_db"])):
                b, v = self.bars[key]; db = max(-90.0, float(db))
                self._bar(b, meter_val(db), RED if db > -3 else GREEN)
                self._upd(v, text=f"{db:.0f} dB")
            b, v = self.bars["vad"]; self._bar(b, m["vad"]); self._upd(v, text=f"{m['vad']*100:.0f}%")
            self._upd(self.gate_lbl, text="open" if m["gate_open"] else "shut", text_color=GREEN if m["gate_open"] else DIM)
            self._upd(self.exp_lbl, text=f"expander {m['exp_db']:.0f} dB")
            if self.eng.get()["denoiser"] == "dfn":
                st = m["dfn"]
                txt, col = {"ready": ("DFN3 ready", GREEN), "loading": ("DFN3 loading...", "#d9822b")}.get(st, (None, RED))
                if txt is None: txt = "DFN3 off - using RNNoise (" + st.split(":")[0] + ")"
            else:
                txt, col = "", DIM
            self._upd(self.dfn_lbl, text=txt, text_color=col)
            self._upd(self.lat_lbl, text=(f"latency ~{m['latency_ms']:.0f} ms   |   glitches {m['glitches']}" if m["running"] else ""),
                      text_color=RED if m["running"] and m["glitches"] else DIM)
        if mini_vis: self.mini_tick(m)
        self.after(100, self._tick)
    # ------------------------------------------------------------------ mini widget
    def show_mini(self):
        if self.mini and self.mini.winfo_exists():
            self.mini.deiconify(); self.mini.lift(); return
        w = self.mini = ctk.CTkToplevel(self)
        w.overrideredirect(True); w.attributes("-topmost", True); w.configure(fg_color=CARD)
        w.geometry("260x182+60+60")
        bar = ctk.CTkFrame(w, fg_color="#0a0a0a", height=26, corner_radius=0); bar.pack(fill="x")
        t = ctk.CTkLabel(bar, text="VOICE CHAIN", text_color=GREEN, font=ctk.CTkFont(size=11, weight="bold")); t.pack(side="left", padx=8)
        ctk.CTkButton(bar, text="x", width=22, height=20, fg_color="transparent", hover_color="#5a1f22", command=w.withdraw).pack(side="right", padx=3)
        ctk.CTkButton(bar, text="open", width=40, height=20, fg_color="transparent", hover_color="#333", command=self.show).pack(side="right")
        for wid in (bar, t):
            wid.bind("<ButtonPress-1>", lambda e: setattr(w, "_d", (e.x_root - w.winfo_x(), e.y_root - w.winfo_y())))
            wid.bind("<B1-Motion>", lambda e: w.geometry(f"+{e.x_root - w._d[0]}+{e.y_root - w._d[1]}"))
        self.m_power = ctk.CTkSwitch(w, text="ON / OFF", progress_color=GREEN, command=self._mini_power); self.m_power.pack(anchor="w", padx=12, pady=(8, 2))
        r = ctk.CTkFrame(w, fg_color="transparent"); r.pack(fill="x", padx=12)
        ctk.CTkLabel(r, text="Strength", width=56, anchor="w", text_color=DIM).pack(side="left")
        self.m_str = ctk.CTkSlider(r, from_=0, to=3, number_of_steps=60, progress_color=GREEN, button_color=GREEN,
                                   command=lambda v: self.sliders["exp_strength"].set(v) or self._slide("exp_strength", v))
        self.m_str.pack(side="left", fill="x", expand=True)
        r2 = ctk.CTkFrame(w, fg_color="transparent"); r2.pack(fill="x", padx=12, pady=(4, 0))
        ctk.CTkLabel(r2, text="Volume", width=56, anchor="w", text_color=DIM).pack(side="left")
        self.m_vol = ctk.CTkSlider(r2, from_=-40, to=6, number_of_steps=92, progress_color=GREEN, button_color=GREEN,
                                   command=lambda v: self.sliders["out_db"].set(v) or self._slide("out_db", v))
        self.m_vol.pack(side="left", fill="x", expand=True)
        self.m_bar = ctk.CTkProgressBar(w, progress_color=GREEN, fg_color="#2a2a2a"); self.m_bar.set(0); self.m_bar.pack(fill="x", padx=12, pady=10)
        self.mini_sync()

    def _mini_power(self):
        (self.power.select if self.m_power.get() else self.power.deselect)(); self._toggle_power()

    def mini_sync(self):
        if not (self.mini and self.mini.winfo_exists()): return
        (self.m_power.select if self.eng.running else self.m_power.deselect)()
        self.m_str.set(self.eng.get()["exp_strength"]); self.m_vol.set(self.eng.get()["out_db"])

    def mini_tick(self, m):
        self._bar(self.m_bar, meter_val(max(-90.0, float(m["out_db"]))))

    # ------------------------------------------------------------------ tray
    def _build_tray(self):
        post = lambda fn: (lambda *a: self.cmds.put(fn))
        menu = pystray.Menu(
            pystray.MenuItem("Open dashboard", post(self.show), default=True),
            pystray.MenuItem("Mini widget", post(self.show_mini)),
            pystray.MenuItem("Turn on / off", post(self._tray_toggle)),
            pystray.MenuItem("Quit", post(self.quit_app)))
        self.tray = pystray.Icon("voicechain", tray_image(False), "Voice Chain", menu)
        self.tray.run_detached()

    def _tray_toggle(self):
        (self.power.deselect if self.power.get() else self.power.select)(); self._toggle_power()

    def hide(self): self.withdraw()
    def show(self): self.deiconify(); self.lift(); self.focus_force()

    def quit_app(self):
        self._save_now(); self.eng.shutdown()
        try: self.tray.stop()
        except Exception: pass
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
