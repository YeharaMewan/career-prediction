import { motion } from "framer-motion";
import { useEffect } from "react";

interface SplashScreenProps {
  onComplete: () => void;
}

const SplashScreen = ({ onComplete }: SplashScreenProps) => {
  useEffect(() => {
    // Auto-dismiss after 2 seconds
    const timer = setTimeout(() => {
      onComplete();
    }, 2000);

    return () => clearTimeout(timer);
  }, [onComplete]);

  return (
    <motion.div
      role="status"
      aria-live="polite"
      aria-label="Loading Career.AI application"
      className="fixed inset-0 z-[100] flex items-center justify-center bg-gradient-to-br from-[#22577a] via-[#38a3a5] to-[#57cc99]"
      initial={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.6, ease: "easeInOut" }}
    >
      <span className="sr-only">Loading Career.AI, please wait...</span>

      {/* Animated background orbs */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <motion.div
          className="absolute -top-20 -left-20 h-96 w-96 rounded-full bg-[#80ed99]/30 blur-3xl"
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.3, 0.5, 0.3]
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
        <motion.div
          className="absolute -bottom-20 -right-20 h-96 w-96 rounded-full bg-[#c7f9cc]/30 blur-3xl"
          animate={{
            scale: [1.2, 1, 1.2],
            opacity: [0.5, 0.3, 0.5]
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
      </div>

      {/* Main heading */}
      <motion.div
        className="relative z-10 text-center"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
      >
        <h1 className="font-space-grotesk text-7xl font-bold text-white md:text-8xl lg:text-9xl">
          Career.AI
        </h1>

        {/* Tagline */}
        <motion.p
          className="mt-4 font-jetbrains-mono text-lg text-[#c7f9cc] md:text-xl"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.6 }}
        >
          Building Your Future
        </motion.p>

        {/* Loading progress bar */}
        <motion.div
          className="mx-auto mt-8 h-1 w-32 rounded-full bg-white/30"
          initial={{ width: 0 }}
          animate={{ width: 128 }}
          transition={{ duration: 1.8, ease: "easeInOut" }}
        >
          <motion.div
            className="h-full rounded-full bg-white"
            initial={{ width: "0%" }}
            animate={{ width: "100%" }}
            transition={{ duration: 1.8, ease: "easeInOut" }}
          />
        </motion.div>
      </motion.div>
    </motion.div>
  );
};

export default SplashScreen;
