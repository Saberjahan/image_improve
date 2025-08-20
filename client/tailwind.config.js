   /** @type {import('tailwindcss').Config} */
   export default {
     content: [
       // This tells Tailwind to scan all JS, JSX, TS, TSX files in the src directory.
       // All HTML/JSX containing Tailwind classes MUST be within these files.
       // Files in 'public/' are NOT scanned for Tailwind classes.
       "./src/**/*.{js,jsx,ts,tsx}",
     ],
     theme: {
       extend: {},
     },
     plugins: [],
   }
   