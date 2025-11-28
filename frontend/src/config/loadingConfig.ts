export const LOADING_CONFIG = {
  // Splash screen timings
  splash: {
    appReadyDelay: 500,      // Wait for app initialization
    displayDuration: 2000,   // How long to show splash
    fadeOutDuration: 600,    // Exit animation duration
  },

  // Page skeleton timings
  skeleton: {
    landing: 800,
    faq: 600,
    prediction: 700,
    fadeInDuration: 400,
  },

  // Animation configs
  animations: {
    reducedMotion: typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  },
};
