

import Home from "./pages/Home";
import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import FileAnalysis from "./pages/FileAnalysis";
import Reports from "./pages/Reports";
import SectionHeader from "./components/SectionHeader";
import React, { useState } from "react";
import NewAnalysisModal from "./components/NewAnalysisModal";


const App: React.FC = () => {
	const [page, setPage] = useState<string>("dashboard");
	const [lastFilePage, setLastFilePage] = useState<string>("file1");
	const [fileCount, setFileCount] = useState<number>(4); // next file index (existing files: 1,2,3,$)
	const [isNewModalOpen, setIsNewModalOpen] = useState<boolean>(false);
	const [pages, setPages] = useState<{ key: string; label: string; icon?: string }[]>([
		{ key: 'dashboard', label: 'HOME PAGE' },
		{ key: 'file2', label: 'FILE ANALYSIS 2' },
		{ key: 'file1', label: 'FILE ANALYSIS 1' },
		{ key: 'file3', label: 'FILE ANALYSIS 3' },
		{ key: 'file$', label: 'FILE ANALYSIS $' },
	]);

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

	const renamePage = (key: string) => {
		const current = pages.find(p => p.key === key);
		const currentLabel = current ? current.label : '';
		const name = prompt('Rename page', currentLabel || 'File Analysis');
		if (!name) return;
		setPages((p) => p.map(x => x.key === key ? { ...x, label: name } : x));
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

		content = <FileAnalysis label={pageLabel} onNavigate={navigate} currentPage={page} />;
	} else {
		content = <Home />;
	}

		return (
			<div className="app-layout">
				{/* if we're on reports, keep the sidebar highlighting the last visited file */}
				<Sidebar active={page === 'reports' ? lastFilePage : page} onNavigate={navigate} onAddPage={addPage} onDeletePage={deletePage} onBulkDelete={deletePages} onRenamePage={renamePage} onReorder={reorderPages} pages={pages} />
				<NewAnalysisModal isOpen={isNewModalOpen} defaultName={`FILE ANALYSIS ${fileCount}`} onClose={() => setIsNewModalOpen(false)} onCreate={createPage} />
				<main className="main-content">
				{ /* determine header title: if viewing a file page, show its label; otherwise show Dashboard */ }
			<SectionHeader currentPage={page} onNavigate={navigate} lastFilePage={lastFilePage} title={
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
