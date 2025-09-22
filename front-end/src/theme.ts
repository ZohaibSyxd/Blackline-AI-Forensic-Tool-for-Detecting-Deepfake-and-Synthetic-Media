// Lightweight theme manager: supports 'light' | 'dark' | 'system'
export type ThemeMode = 'light' | 'dark' | 'system';

const STORAGE_KEY = 'bl_theme'; // replaces legacy 'bl_dark'
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
  const current = getStoredTheme();
  applyTheme(current);
  let dispose: (() => void) | undefined;
  if (current === 'system') {
    dispose = watchSystemTheme(() => applyTheme('system'));
  }
  return { mode: current, dispose };
}
