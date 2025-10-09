// Lightweight theme manager: supports theme mode (light/dark/system) and color schemes (accent palettes)
export type ThemeMode = 'light' | 'dark' | 'system';
export type ColorScheme =
  | 'ocean'    // #3D5A80
  | 'emerald'  // #10B981
  | 'indigo'   // #4F46E5
  | 'violet'   // #8B5CF6
  | 'rose'     // #F43F5E
  | 'amber'    // #F59E0B
  | 'slate'    // #64748B
  | 'sky';     // #0EA5E9

const STORAGE_KEY = 'bl_theme'; // replaces legacy 'bl_dark'
const SCHEME_KEY = 'bl_color_scheme';
const LEGACY_DARK_KEY = 'bl_dark';

export function getStoredTheme(): ThemeMode {
  try {
    const v = localStorage.getItem(STORAGE_KEY) as ThemeMode | null;
    if (v === 'light' || v === 'dark' || v === 'system') return v;
    // migrate legacy boolean flag if present
    const legacy = localStorage.getItem(LEGACY_DARK_KEY);
    if (legacy === '1' || legacy === '0') {
      const mode: ThemeMode = legacy === '1' ? 'dark' : 'light';
      localStorage.setItem(STORAGE_KEY, mode);
      return mode;
    }
  } catch {}
  return 'system';
}

export function storeTheme(mode: ThemeMode) {
  try { localStorage.setItem(STORAGE_KEY, mode); } catch {}
}

export function systemPrefersDark(): boolean {
  return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
}

export function applyTheme(mode: ThemeMode) {
  const root = document.documentElement;
  const isDark = mode === 'dark' || (mode === 'system' && systemPrefersDark());
  root.classList.toggle('dark', isDark);
}

export function watchSystemTheme(callback: () => void) {
  const mq = window.matchMedia('(prefers-color-scheme: dark)');
  const handler = () => callback();
  if ('addEventListener' in mq) mq.addEventListener('change', handler);
  else if ('addListener' in mq) (mq as any).addListener(handler);
  return () => {
    if ('removeEventListener' in mq) mq.removeEventListener('change', handler);
    else if ('removeListener' in mq) (mq as any).removeListener(handler);
  };
}

export function initTheme() {
  const mode = getStoredTheme();
  applyTheme(mode);
  // Apply stored color scheme (accent palette)
  const scheme = getStoredColorScheme();
  applyColorScheme(scheme);
  let dispose: undefined | (() => void);
  if (mode === 'system') {
    dispose = watchSystemTheme(() => applyTheme('system'));
  }
  return { mode, dispose };
}

// ---------------- Color scheme management ----------------
const SCHEMES: Record<ColorScheme, { label: string; rgb: [number, number, number] }> = {
  ocean:   { label: 'Ocean',   rgb: [61, 90, 128] },  // #3D5A80
  emerald: { label: 'Emerald', rgb: [16, 185, 129] }, // #10B981
  indigo:  { label: 'Indigo',  rgb: [79, 70, 229] },  // #4F46E5
  violet:  { label: 'Violet',  rgb: [139, 92, 246] }, // #8B5CF6
  rose:    { label: 'Rose',    rgb: [244, 63, 94] },  // #F43F5E
  amber:   { label: 'Amber',   rgb: [245, 158, 11] }, // #F59E0B
  slate:   { label: 'Slate',   rgb: [100, 116, 139] },// #64748B
  sky:     { label: 'Sky',     rgb: [14, 165, 233] }, // #0EA5E9
};

export function listColorSchemes(): Array<{ key: ColorScheme; label: string; rgb: [number, number, number] }>{
  return (Object.keys(SCHEMES) as ColorScheme[]).map(k => ({ key: k, label: SCHEMES[k].label, rgb: SCHEMES[k].rgb }));
}

export function getStoredColorScheme(): ColorScheme {
  try {
    const v = localStorage.getItem(SCHEME_KEY) as ColorScheme | null;
    if (v && (v in SCHEMES)) return v;
  } catch {}
  return 'ocean';
}

export function storeColorScheme(scheme: ColorScheme) {
  try { localStorage.setItem(SCHEME_KEY, scheme); } catch {}
}

export function applyColorScheme(scheme: ColorScheme) {
  const root = document.documentElement;
  const rgb = SCHEMES[scheme]?.rgb || SCHEMES['ocean'].rgb;
  root.style.setProperty('--accent-rgb', `${rgb[0]}, ${rgb[1]}, ${rgb[2]}`);
  // Foreground stays white by default; users can override via CSS if needed.
}
