

import Home from "./pages/Home";
import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import FileAnalysis from "./pages/FileAnalysis";
import Reports from "./pages/Reports";
import SectionHeader from "./components/SectionHeader";
import React, { useEffect, useState } from "react";
import { refreshUser } from './state/authStore';
import NewAnalysisModal from "./components/NewAnalysisModal";


const App: React.FC = () => {
	const STORAGE = {
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

	const loadPages = (): { key: string; label: string; icon?: string }[] => {
		try {
			const raw = localStorage.getItem(STORAGE.pages);
			if (!raw) return defaultPages;
			const parsed = JSON.parse(raw);
			if (!Array.isArray(parsed)) return defaultPages;
			const cleaned = parsed.filter((p: any) => p && typeof p.key === 'string' && typeof p.label === 'string')
								  .map((p: any) => ({ key: p.key, label: p.label, icon: p.icon }));
			// ensure dashboard exists at index 0
			const hasDash = cleaned.some(p => p.key === 'dashboard');
			const withoutDash = cleaned.filter(p => p.key !== 'dashboard');
			return [{ key: 'dashboard', label: 'HOME PAGE' }, ...(hasDash ? withoutDash : withoutDash)];
		} catch { return defaultPages; }
	};

	const [pages, setPages] = useState<{ key: string; label: string; icon?: string }[]>(() => loadPages());
	const [page, setPage] = useState<string>(() => localStorage.getItem(STORAGE.page) || 'dashboard');
	const [lastFilePage, setLastFilePage] = useState<string>(() => localStorage.getItem(STORAGE.last) || 'file1');
	const [fileCount, setFileCount] = useState<number>(() => {
		const raw = localStorage.getItem(STORAGE.count);
		const n = raw ? parseInt(raw, 10) : 4;
		return Number.isFinite(n) && n > 0 ? n : 4;
	});
	const [isNewModalOpen, setIsNewModalOpen] = useState<boolean>(false);

	// Restore auth session (if token present) on first mount
	useEffect(() => { try { refreshUser(); } catch {} }, []);

	// Persist to localStorage when these values change
	useEffect(() => { try { localStorage.setItem(STORAGE.pages, JSON.stringify(pages)); } catch {} }, [pages]);
	useEffect(() => { try { localStorage.setItem(STORAGE.page, page); } catch {} }, [page]);
	useEffect(() => { try { localStorage.setItem(STORAGE.last, lastFilePage); } catch {} }, [lastFilePage]);
	useEffect(() => { try { localStorage.setItem(STORAGE.count, String(fileCount)); } catch {} }, [fileCount]);

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
		content = <Dashboard />;
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
				{ /* determine header title: if viewing a file page, show its label; otherwise show Dashboard */ }
			<SectionHeader currentPage={page} onNavigate={navigate} lastFilePage={lastFilePage} onDeleteCurrent={deletePage} title={
				page.startsWith('file')
				    ? (pages.find(p => p.key === page)?.label || 'File Analysis')
				    : (page === 'reports' ? (pages.find(p => p.key === lastFilePage)?.label || 'Reports') : 'Dashboard')
			    } />
					{content}
					</main>
			</div>
		);
};

export default App;
