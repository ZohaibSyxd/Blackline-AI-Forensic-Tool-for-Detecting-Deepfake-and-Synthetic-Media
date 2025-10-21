

import Home from "./pages/Home";
import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import FileAnalysis from "./pages/FileAnalysis";
import Reports from "./pages/Reports";
import SectionHeader from "./components/SectionHeader";
import React, { useEffect, useState } from "react";
import { refreshUser, subscribe as authSubscribe, getAuthState } from './state/authStore';
import NewAnalysisModal from "./components/NewAnalysisModal";
import { getPages as apiGetPages, putPages as apiPutPages } from './utils/assetsApi';


const App: React.FC = () => {
	// Auth state subscription (so we can scope pages per user)
	const [auth, setAuth] = useState(getAuthState());
	useEffect(() => { const unsub = authSubscribe(setAuth); return () => { unsub(); }; }, []);

	const STORAGE_BASE = {
		// legacy single-key pages stored as array: 'bl_pages' or 'bl_pages_guest'
		pages: 'bl_pages',
		// new: map of username -> pages array
		pagesByUser: 'bl_pages_by_user',
		page: 'bl_page',
		last: 'bl_lastFilePage',
	} as const;

	const DASHBOARD_ONLY: { key: string; label: string; icon?: string }[] = [
		{ key: 'dashboard', label: 'HOME PAGE' },
	];

	function userNamespace() { return auth.user ? `u_${auth.user.username}` : 'guest'; }
	function keyFor(base: keyof typeof STORAGE_BASE) { return `${STORAGE_BASE[base]}_${userNamespace()}`; }

	const loadPages = (ns?: string): { key: string; label: string; icon?: string }[] => {
		const namespace = ns || userNamespace();
		try {
			// First try new per-user map
			const rawMap = localStorage.getItem(STORAGE_BASE.pagesByUser);
			if (rawMap) {
				const parsedMap = JSON.parse(rawMap || '{}');
				const arr = parsedMap && parsedMap[namespace];
				if (Array.isArray(arr)) {
					// If user intentionally deleted all pages, persist empty array and only show dashboard
					const cleaned = arr.filter((p: any) => p && typeof p.key === 'string' && typeof p.label === 'string')
							.map((p: any) => ({ key: p.key, label: p.label, icon: p.icon }));
					if (cleaned.length === 0) {
						return [...DASHBOARD_ONLY];
					}
					const hasDash = cleaned.some(p => p.key === 'dashboard');
					const withoutDash = cleaned.filter(p => p.key !== 'dashboard');
					return [...DASHBOARD_ONLY, ...(hasDash ? withoutDash : withoutDash)];
				}
				// If a per-user pages map exists and entry is present but empty, try migration from guest
				if (Array.isArray(arr) && arr.length === 0 && parsedMap && Array.isArray(parsedMap['guest']) && parsedMap['guest'].length > 0) {
					const guestCleaned = (parsedMap['guest'] as any[])
						.filter((p: any) => p && typeof p.key === 'string' && typeof p.label === 'string')
						.map((p: any) => ({ key: p.key, label: p.label, icon: p.icon }));
					parsedMap[namespace] = guestCleaned;
					try { delete parsedMap['guest']; } catch {}
					try { localStorage.setItem(STORAGE_BASE.pagesByUser, JSON.stringify(parsedMap)); } catch {}
					const hasDash = guestCleaned.some(p => p.key === 'dashboard');
					const withoutDash = guestCleaned.filter(p => p.key !== 'dashboard');
					return [...DASHBOARD_ONLY, ...(hasDash ? withoutDash : withoutDash)];
				}
				// If a per-user pages map exists but there's no entry for this namespace,
				// try to migrate any existing guest pages to this new user namespace
				// (covers the case where pages were created before login).
				if (parsedMap && Array.isArray(parsedMap['guest']) && parsedMap['guest'].length > 0) {
					const guestCleaned = (parsedMap['guest'] as any[])
						.filter((p: any) => p && typeof p.key === 'string' && typeof p.label === 'string')
						.map((p: any) => ({ key: p.key, label: p.label, icon: p.icon }));
					parsedMap[namespace] = guestCleaned;
					try { delete parsedMap['guest']; } catch {}
					try { localStorage.setItem(STORAGE_BASE.pagesByUser, JSON.stringify(parsedMap)); } catch {}
					const hasDash = guestCleaned.some(p => p.key === 'dashboard');
					const withoutDash = guestCleaned.filter(p => p.key !== 'dashboard');
					return [...DASHBOARD_ONLY, ...(hasDash ? withoutDash : withoutDash)];
				}
				// treat this as an intentional "no pages created yet" state instead of
				// falling back to the full `defaultPages` which would re-create pages
				// the user deleted. Persist an empty array for this namespace and
				// return only the dashboard entry.
				try {
					parsedMap[namespace] = [];
					localStorage.setItem(STORAGE_BASE.pagesByUser, JSON.stringify(parsedMap));
				} catch {}
				return [...DASHBOARD_ONLY];
			}
			// Fallback to legacy keys (namespaced or global)
			const storageKey = `${STORAGE_BASE.pages}_${namespace}`;
			let raw = localStorage.getItem(storageKey);
			if (!raw && namespace === 'guest') raw = localStorage.getItem(STORAGE_BASE.pages); // legacy
			if (!raw) return [...DASHBOARD_ONLY];
			const parsed = JSON.parse(raw);
			if (!Array.isArray(parsed)) return [...DASHBOARD_ONLY];
			const cleaned = parsed.filter((p: any) => p && typeof p.key === 'string' && typeof p.label === 'string')
								  .map((p: any) => ({ key: p.key, label: p.label, icon: p.icon }));
			// Migrate legacy array into new per-user map so future loads use the namespaced storage.
			try { savePagesForUser(cleaned, namespace); } catch {}
			// Remove legacy keys to avoid resurrecting deleted pages on other flows
			try { localStorage.removeItem(storageKey); if (namespace === 'guest') localStorage.removeItem(STORAGE_BASE.pages); } catch {}
			if (cleaned.length === 0) {
				return [...DASHBOARD_ONLY];
			}
			const hasDash = cleaned.some(p => p.key === 'dashboard');
			const withoutDash = cleaned.filter(p => p.key !== 'dashboard');
			return [...DASHBOARD_ONLY, ...(hasDash ? withoutDash : withoutDash)];
		} catch (e) { console.error('[bl] loadPages error', e); return [{ key: 'dashboard', label: 'HOME PAGE' }]; }
	};

	// Save pages into the new per-user map (migrate legacy if needed)
	const savePagesForUser = (pagesToSave: { key: string; label: string; icon?: string }[], ns?: string) => {
		const namespace = ns || userNamespace();
		try {
			let map: Record<string, any> = {};
			const rawMap = localStorage.getItem(STORAGE_BASE.pagesByUser);
			if (rawMap) {
				map = JSON.parse(rawMap) || {};
			}
			map[namespace] = pagesToSave;
			localStorage.setItem(STORAGE_BASE.pagesByUser, JSON.stringify(map));
			// Clean up legacy keys so old storage doesn't resurrect deleted pages.
			try {
				localStorage.removeItem(`${STORAGE_BASE.pages}_${namespace}`);
				if (namespace === 'guest') localStorage.removeItem(STORAGE_BASE.pages);
			} catch {}
		} catch {}
	};

	// Core page state (start with no file pages; then hydrate per user)
	const [pages, setPages] = useState<{ key: string; label: string; icon?: string }[]>([...DASHBOARD_ONLY]);
	const [page, setPage] = useState<string>('dashboard');
	const [lastFilePage, setLastFilePage] = useState<string>('file1');
	// helper to compute next available file index (1-based)
	const computeNextIndex = (arr: { key: string; label: string }[]) => {
		const used = new Set(
			arr.map(p => {
				const m = /^file(\d+)$/.exec(p.key);
				return m ? parseInt(m[1], 10) : NaN;
			}).filter(n => Number.isFinite(n) && n > 0) as number[]
		);
		let n = 1;
		while (used.has(n)) n += 1;
		return n;
	};

	// Hydrate (or switch) when user changes
	useEffect(() => {
		const ns = userNamespace();
		const loadedLocal = loadPages(ns);
		// Try server if authenticated; else use local
		(async () => {
			let effective = loadedLocal;
			const isAuthed = !!auth.user && !!auth.token;
			if (isAuthed) {
				try {
					const remote = await apiGetPages();
					if (remote && Array.isArray(remote)) {
						if (remote.length === 0 && loadedLocal.filter(p=>p.key!=='dashboard').length > 0) {
							// First-time migration: push local pages to backend
							await apiPutPages(loadedLocal.filter(p=>p.key!=='dashboard'));
							effective = loadedLocal;
						} else if (remote.length > 0) {
							effective = [{ key: 'dashboard', label: 'HOME PAGE' }, ...remote.filter(p=>p.key!=='dashboard')];
						}
					}
				} catch {}
			}
			setPages(effective);
			// After pages resolved, restore last viewed page (only if it exists for this user)
			const savedPage = localStorage.getItem(keyFor('page')) || 'dashboard';
			const allowed = savedPage === 'reports' || effective.some((p: { key: string }) => p.key === savedPage);
			setPage(allowed ? savedPage : 'dashboard');
			const savedLast = localStorage.getItem(keyFor('last')) || 'file1';
			setLastFilePage(savedLast);
			// Also ensure local mirror stays up to date for offline
			try { savePagesForUser(effective); } catch {}
		})();
	}, [auth.user]);
	const [isNewModalOpen, setIsNewModalOpen] = useState<boolean>(false);

	// Restore auth session (if token present) on first mount
	useEffect(() => { try { refreshUser(); } catch {} }, []);

	// Global navigation listener so components (e.g., UploadCard) can request page changes
	useEffect(() => {
		function onNav(ev: Event) {
			try {
				const ce = ev as CustomEvent<{ page: string; filePage?: string }>;
				const target = ce.detail?.page;
				if (!target) return;
				if (target.startsWith('file')) {
					setLastFilePage(target);
					setPage(target);
				} else if (target === 'reports') {
					setPage('reports');
				}
			} catch {}
		}
		window.addEventListener('bl:navigate', onNav as EventListener);
		return () => { window.removeEventListener('bl:navigate', onNav as EventListener); };
	}, []);

	// Persist to localStorage when these values change
	useEffect(() => { try { savePagesForUser(pages); } catch {} }, [pages, auth.user]);

	// Helper: push pages to backend only on user-initiated changes
	const pushPages = async (arr: { key: string; label: string; icon?: string }[]) => {
		const isAuthed = !!auth.user && !!auth.token;
		if (!isAuthed) return;
		try { await apiPutPages(arr.filter(p => p.key !== 'dashboard')); } catch {}
	};
	useEffect(() => { try { localStorage.setItem(keyFor('page'), page); } catch {} }, [page, auth.user]);
	useEffect(() => { try { localStorage.setItem(keyFor('last'), lastFilePage); } catch {} }, [lastFilePage, auth.user]);

	// If current page no longer exists (e.g., deleted), fall back to dashboard
	useEffect(() => {
		if (page.startsWith('file') && !pages.some(p => p.key === page)) {
			setPage('dashboard');
		}
	}, [pages, page]);

	const navigate = (p: string) => {
		if (p.startsWith('file')) {
			setLastFilePage(p);
			setPage(p);
		} else {
			setPage(p);
		}
	};

	const addPage = () => setIsNewModalOpen(true);

	const createPage = (name: string, icon?: string) => {
		setPages((prev) => {
			const nextIdx = computeNextIndex(prev);
			const newKey = `file${nextIdx}`;
			const next = [...prev, { key: newKey, label: name, icon }];
			// Persist immediately to avoid any refresh race
			try { savePagesForUser(next); } catch {}
			// Push to backend if authenticated
			pushPages(next);
			// navigate to the new file page
			setIsNewModalOpen(false);
			setLastFilePage(newKey);
			setPage(newKey);
			return next;
		});
	};
//basic
			const deletePage = (key: string) => {
				// Update state and persist immediately to avoid a race where a refresh
				// happens before the effect flushes and resurrects deleted pages.
				setPages((prev) => {
					const next = prev.filter((x) => x.key !== key);
					try { savePagesForUser(next); } catch {}
					pushPages(next);
					return next;
				});
				// if the deleted page was the lastFilePage, switch to the first available file page (if any)
				setLastFilePage((prev) => {
					if (prev !== key) return prev;
					// choose the first remaining file page
					const remaining = pages.filter(p => p.key !== 'dashboard' && p.key !== key && p.key.startsWith('file'));
					return remaining.length ? remaining[0].key : 'file1';
				});
				// clear persisted current page if it pointed at the deleted page
				try {
					const cur = localStorage.getItem(keyFor('page'));
					if (cur === key) localStorage.setItem(keyFor('page'), 'dashboard');
				} catch {}
				// if we're currently viewing the deleted page, navigate to dashboard
				if (page === key) navigate('dashboard');
			};

		// Bulk delete multiple pages by key
			const deletePages = (keys: string[]) => {
				if (!keys || keys.length === 0) return;
				const keySet = new Set(keys);
				setPages((prev) => {
					const next = prev.filter((x) => !keySet.has(x.key));
					try { savePagesForUser(next); } catch {}
					pushPages(next);
					return next;
				});
				// if any deleted page was the lastFilePage, switch to first available file page (if any)
				setLastFilePage((prev) => {
					if (prev && keySet.has(prev)) {
						const remaining = pages.filter(p => p.key !== 'dashboard' && !keySet.has(p.key) && p.key.startsWith('file'));
						return remaining.length ? remaining[0].key : 'file1';
					}
					return prev;
				});
				// clear persisted current page if it was among deleted
				try {
					const cur = localStorage.getItem(keyFor('page'));
					if (cur && keySet.has(cur)) localStorage.setItem(keyFor('page'), 'dashboard');
				} catch {}
				// if we're currently viewing a deleted page, navigate to dashboard
				if (keySet.has(page)) navigate('dashboard');
			};

	// Reorder pages (only reorders non-dashboard items).
	const reorderPages = (fromKey: string, toIndex: number) => {
		setPages((prev) => {
			const dashboard = prev.find(p => p.key === 'dashboard');
			const others = prev.filter(p => p.key !== 'dashboard');
			const fromIdx = others.findIndex(p => p.key === fromKey);
			if (fromIdx === -1) return prev; // nothing to do
			const item = others.splice(fromIdx, 1)[0];
			// clamp toIndex
			if (toIndex < 0) toIndex = 0;
			if (toIndex > others.length) toIndex = others.length;
			others.splice(toIndex, 0, item);
			const next = dashboard ? [dashboard, ...others] : [{ key: 'dashboard', label: 'HOME PAGE' }, ...others];
			try { savePagesForUser(next); } catch {}
			pushPages(next);
			return next;
		});
	};

	const renamePage = (key: string, newName?: string) => {
		// If a new name is provided (inline rename), use it; otherwise fall back to prompt.
		let name = (newName !== undefined) ? newName : undefined;
		if (name === undefined) {
			const current = pages.find(p => p.key === key);
			const currentLabel = current ? current.label : '';
			name = prompt('Rename page', currentLabel || 'File Analysis') || undefined;
		}
		if (!name) return;
		setPages((p) => {
			const next = p.map(x => x.key === key ? { ...x, label: name as string } : x);
			try { savePagesForUser(next); } catch {}
			pushPages(next);
			return next;
		});
	};

	const changeIcon = (key: string, icon?: string) => {
		setPages(p => {
			const next = p.map(x => x.key === key ? { ...x, icon } : x);
			try { savePagesForUser(next); } catch {}
			pushPages(next);
			return next;
		});
	};

	let content;
	if (page === "dashboard") {
		content = <Dashboard pages={pages} />;
	} else if (page === "reports") {
		content = <Reports filePage={lastFilePage} />;
	} else if (page.startsWith("file")) {
		// derive label from pages state (covers dynamically added pages)
		const pageEntry = pages.find((p) => p.key === page);
		let pageLabel = pageEntry ? pageEntry.label : null;
		if (!pageLabel) {
			// fallback: parse number from key, e.g. file4 -> FILE ANALYSIS 4
			const m = page.match(/^file(\d+)$/);
			pageLabel = m ? `FILE ANALYSIS ${m[1]}` : "FILE ANALYSIS";
		}

	content = <FileAnalysis key={page} label={pageLabel} onNavigate={navigate} currentPage={page} />;
	} else {
		content = <Home />;
	}

		return (
			<div className="app-layout">
				{/* if we're on reports, keep the sidebar highlighting the last visited file */}
				<Sidebar active={page === 'reports' ? lastFilePage : page} onNavigate={navigate} onAddPage={addPage} onDeletePage={deletePage} onBulkDelete={deletePages} onRenamePage={renamePage} onReorder={reorderPages} onChangeIcon={changeIcon} pages={pages} />
				{/* Compute next label dynamically so first page starts at 1 and fills gaps */}
				<NewAnalysisModal isOpen={isNewModalOpen} defaultName={`FILE ANALYSIS ${computeNextIndex(pages)}`} onClose={() => setIsNewModalOpen(false)} onCreate={createPage} />
				<main className="main-content">
				{ /* determine header title and icon: if viewing a file page, show its label and icon; for reports, use last file page's label and icon; otherwise Dashboard with no icon */ }
			<SectionHeader
				currentPage={page}
				onNavigate={navigate}
				lastFilePage={lastFilePage}
				onDeleteCurrent={deletePage}
				onChangeIcon={changeIcon}
				title={
					page.startsWith('file')
					    ? (pages.find(p => p.key === page)?.label || 'File Analysis')
					    : (page === 'reports' ? (pages.find(p => p.key === lastFilePage)?.label || 'Reports') : 'Dashboard')
				}
				icon={
					page.startsWith('file')
					  ? (pages.find(p => p.key === page)?.icon || '📁')
					  : (page === 'reports' ? (pages.find(p => p.key === lastFilePage)?.icon || '📁') : undefined)
				}
			/>
					{content}
					</main>
			</div>
		);
};

export default App;
