/** @type {import('tailwindcss').Config} */
module.exports = {
	prefix: 'tw-',
	important: false,
	content: [
		"**/*.{html, jsx, js}",
		"**/*.js",
		"**/*.html",
	],
	darkMode: 'class',
	theme: {
		extend: {
			colors: {
				primary: "#c490ff",
				secondary: "#6c72e8", 
			}
		},
	},
	plugins: [],
}


// Use this to build css classes: npx tailwindcss -i ./static/styles/input.css -o ./static/styles/main.css --watch