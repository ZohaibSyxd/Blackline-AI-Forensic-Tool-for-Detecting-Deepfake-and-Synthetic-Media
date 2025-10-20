

import Home from "./pages/Home";
import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import FileAnalysis from "./pages/FileAnalysis";
import Reports from "./pages/Reports";
import SectionHeader from "./components/SectionHeader";
import React, { useEffect, useState } from "react";
import { refreshUser, subscribe as authSubscribe, getAuthState } from './state/authStore';
import NewAnalysisModal from "./components/NewAnalysisModal";


const App: React.FC = () => {
	// Auth state subscription (so we can scope pages per user)
	const [auth, setAuth] = useState(getAuthState());
	useEffect(() => { const unsub = authSubscribe(setAuth); return () => { unsub(); }; }, []);

	const STORAGE_BASE = {
		pages: 'bl_pages',
		page: 'bl_page',
		last: 'bl_lastFilePage',
		count: 'bl_fileCount',
	} as const;

	const defaultPages: { key: string; label: string; icon?: string }[] = [
		{ key: 'dashboard', label: 'HOME PAGE' },
		{ key: 'file2', label: 'FILE ANALYSIS 2' },
		{ key: 'file1', label: 'FILE ANALYSIS 1' },
		{ key: 'file3', label: 'FILE ANALYSIS 3' },
		{ key: 'file$', label: 'FILE ANALYSIS $' },
	];

	function userNamespace() { return auth.user ? `u_${auth.user.username}` : 'guest'; }
	function keyFor(base: keyof typeof STORAGE_BASE) { return `${STORAGE_BASE[base]}_${userNamespace()}`; }

	const loadPages = (ns?: string): { key: string; label: string; icon?: string }[] => {
		const namespace = ns || userNamespace();
		try {
			// backward compatibility for pre-namespaced guest data
			const storageKey = `${STORAGE_BASE.pages}_${namespace}`;
			let raw = localStorage.getItem(storageKey);
			if (!raw && namespace === 'guest') raw = localStorage.getItem(STORAGE_BASE.pages); // legacy
			if (!raw) return defaultPages;
			const parsed = JSON.parse(raw);
			if (!Array.isArray(parsed)) return defaultPages;
			const cleaned = parsed.filter((p: any) => p && typeof p.key === 'string' && typeof p.label === 'string')
								  .map((p: any) => ({ key: p.key, label: p.label, icon: p.icon }));
			const hasDash = cleaned.some(p => p.key === 'dashboard');
			const withoutDash = cleaned.filter(p => p.key !== 'dashboard');
			return [{ key: 'dashboard', label: 'HOME PAGE' }, ...(hasDash ? withoutDash : withoutDash)];
		} catch { return defaultPages; }
	};

	// Core page state (initialized empty, then hydrated in effect below)
	const [pages, setPages] = useState<{ key: string; label: string; icon?: string }[]>(defaultPages);
	const [page, setPage] = useState<string>('dashboard');
	const [lastFilePage, setLastFilePage] = useState<string>('file1');
	const [fileCount, setFileCount] = useState<number>(4);

	// Hydrate (or switch) when user changes
	useEffect(() => {
		const ns = userNamespace();
		const loadedPages = loadPages(ns);
		setPages(loadedPages);
		const savedPage = localStorage.getItem(keyFor('page')) || 'dashboard';
		// Ensure saved page exists for this user
		setPage(loadedPages.some(p => p.key === savedPage) ? savedPage : 'dashboard');
		const savedLast = localStorage.getItem(keyFor('last')) || 'file1';
		setLastFilePage(savedLast);
		const rawCount = localStorage.getItem(keyFor('count'));
		const n = rawCount ? parseInt(rawCount, 10) : 4;
		setFileCount(Number.isFinite(n) && n > 0 ? n : 4);
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
	useEffect(() => { try { localStorage.setItem(keyFor('pages'), JSON.stringify(pages)); } catch {} }, [pages, auth.user]);
	useEffect(() => { try { localStorage.setItem(keyFor('page'), page); } catch {} }, [page, auth.user]);
	useEffect(() => { try { localStorage.setItem(keyFor('last'), lastFilePage); } catch {} }, [lastFilePage, auth.user]);
	useEffect(() => { try { localStorage.setItem(keyFor('count'), String(fileCount)); } catch {} }, [fileCount, auth.user]);

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
		const newKey = `file${fileCount}`;
		setFileCount((c) => c + 1);
		setPages((p) => [...p, { key: newKey, label: name, icon }]);
		setIsNewModalOpen(false);
		// navigate to the new file page
		navigate(newKey);
	};
//basic
		const deletePage = (key: string) => {
			setPages((p) => p.filter((x) => x.key !== key));
			// if the deleted page was the lastFilePage, clear lastFilePage to a default
			setLastFilePage((prev) => (prev === key ? 'file1' : prev));
			// if we're currently viewing the deleted page, navigate to dashboard
			if (page === key) navigate('dashboard');
		};

		// Bulk delete multiple pages by key
		const deletePages = (keys: string[]) => {
			if (!keys || keys.length === 0) return;
			const keySet = new Set(keys);
			setPages((p) => p.filter((x) => !keySet.has(x.key)));
			// if any deleted page was the lastFilePage, reset to default
			setLastFilePage((prev) => (prev && keySet.has(prev) ? 'file1' : prev));
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
			return dashboard ? [dashboard, ...others] : [{ key: 'dashboard', label: 'HOME PAGE' }, ...others];
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
		setPages((p) => p.map(x => x.key === key ? { ...x, label: name as string } : x));
	};

	const changeIcon = (key: string, icon?: string) => {
		setPages(p => p.map(x => x.key === key ? { ...x, icon } : x));
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
				<NewAnalysisModal isOpen={isNewModalOpen} defaultName={`FILE ANALYSIS ${fileCount}`} onClose={() => setIsNewModalOpen(false)} onCreate={createPage} />
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
