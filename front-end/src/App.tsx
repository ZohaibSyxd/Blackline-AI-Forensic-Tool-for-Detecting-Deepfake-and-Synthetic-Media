

import Home from "./pages/Home";
import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import FileAnalysis from "./pages/FileAnalysis";
import Reports from "./pages/Reports";
import SectionHeader from "./components/SectionHeader";
import React, { useState } from "react";


const App: React.FC = () => {
	const [page, setPage] = useState<string>("dashboard");
	const [lastFilePage, setLastFilePage] = useState<string>("file1");
	const [fileCount, setFileCount] = useState<number>(4); // next file index (existing files: 1,2,3,$)
	const [pages, setPages] = useState<{ key: string; label: string }[]>([
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

	const addPage = () => {
	const newKey = `file${fileCount}`;
	const newLabel = `FILE ANALYSIS ${fileCount}`;
	setFileCount((c) => c + 1);
	setPages((p) => [...p, { key: newKey, label: newLabel }]);
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
				<Sidebar active={page === 'reports' ? lastFilePage : page} onNavigate={navigate} onAddPage={addPage} onDeletePage={deletePage} onRenamePage={renamePage} pages={pages} />
				<main className="main-content">
				<SectionHeader currentPage={page} onNavigate={navigate} lastFilePage={lastFilePage} />
					{content}
					</main>
			</div>
		);
};

export default App;
