import React from "react";

const Home: React.FC = () => {
	return (
		<main className="min-h-screen flex flex-col items-center justify-center bg-gray-50">
			<div className="max-w-2xl w-full p-8 bg-white rounded-lg shadow-md">
				<h1 className="text-3xl font-bold mb-4 text-center text-gray-900">Welcome to BlacklineAI</h1>
				<p className="text-lg text-gray-700 text-center mb-6">
					This is your forensic tool for detecting deepfake and synthetic media.
				</p>
				<div className="flex justify-center">
									<button className="px-6 py-2 rounded transition btn-accent">
						Get Started
					</button>
				</div>
			</div>
		</main>
	);
};

export default Home;
