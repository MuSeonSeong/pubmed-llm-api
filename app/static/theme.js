// Runs before the stylesheet applies, so a saved light or dark choice never flashes the other theme.
// A classic script file rather than an inline one, so the pages' Content-Security-Policy can forbid inline code.
try {
  const theme = localStorage.getItem('pubmed-chat.theme');
  if (theme === 'light' || theme === 'dark') document.documentElement.dataset.theme = theme;
} catch { /* storage blocked: follow the system theme */ }
