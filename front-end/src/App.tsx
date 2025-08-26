

import Home from "./pages/Home";
import Sidebar from "./components/Sidebar";
import Dashboard from "./pages/Dashboard";
import FileAnalysis from "./pages/FileAnalysis";
import React, { useState } from "react";


const App: React.FC = () => {
	const [page, setPage] = useState<string>("dashboard");

	let content;
	if (page === "dashboard") {
		content = <Dashboard />;
	} else if (page.startsWith("file")) {
		content = <FileAnalysis label={
			page === "file1" ? "FILE ANALYSIS 1" :
			page === "file2" ? "FILE ANALYSIS 2" :
			page === "file3" ? "FILE ANALYSIS 3" :
			page === "file$" ? "FILE ANALYSIS $" : "FILE ANALYSIS"
		} />;
	} else {
		content = <Home />;
	}

	return (
		<div className="app-layout">
			<Sidebar active={page} onNavigate={setPage} />
			{content}
		</div>
	);
};

export default App;
