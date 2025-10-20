import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
	plugins: [react()],
	// When deploying to GitHub Pages with a custom domain (via CNAME), base can remain '/'.
	base: '/',
	server: {
		open: true,
	},
});
