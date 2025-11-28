import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { ModernSimpleInput } from "../../../components/Input";
import LandingSkeleton from "../../../components/skeletons/LandingSkeleton";
import { useLanguage } from "../../../context/LanguageContext";

const Landing = () => {
  const [value, setValue] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const { language, setLanguage } = useLanguage();
  const navigate = useNavigate();

  useEffect(() => {
    // Simulate content loading
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 800);

    return () => clearTimeout(timer);
  }, []);

  const handleStartChat = () => {
    navigate("/chat", { state: { initialMessage: value } });
  };

  if (isLoading) {
    return <LandingSkeleton />;
  }

  return (
    <div className="relative w-full">
      <div className="relative flex w-full h-[calc(100vh-72px)] items-center justify-center px-6">
        <div className="pointer-events-none absolute inset-0 overflow-hidden">
          <div className="absolute top-10 right-0 h-90 w-96 rounded-full bg-teal-300/30 blur-3xl" />
          <div className="absolute bottom-10 left-0 h-90 w-96 rounded-full bg-cyan-400/30 blur-3xl" />
        </div>

        <div className="relative z-10 w-full max-w-5xl text-center">
          <div className="space-y-7 xl:space-y-9">
            <h1 className="text-3xl leading-tight font-bold text-gray-900 md:text-6xl lg:text-7xl xl:text-8xl">
              Build Imagination
              <br />
              <span className="bg-gradient-to-r from-teal-500 to-cyan-600 bg-clip-text text-transparent">
                With Logic AI
              </span>
            </h1>

            <p className="mx-auto max-w-2xl font-mono text-lg font-light text-gray-700 md:text-xl">
              Your intelligent companion for creativity, problem-solving, and turning ideas into
              reality.
            </p>

            {/* Language Toggle */}
            <div className="flex justify-center">
              <button
                onClick={() => setLanguage(language === "en" ? "si" : "en")}
                className="relative inline-flex cursor-pointer rounded-full bg-gray-100/80 p-1 backdrop-blur-sm ring-1 ring-gray-200 transition-all hover:ring-gray-300 focus:outline-none focus:ring-2 focus:ring-teal-500/50"
              >
                <span
                  className={`relative z-10 rounded-full px-4 py-1.5 text-sm font-medium transition-colors duration-300 ${language === "en" ? "text-white" : "text-gray-500"
                    }`}
                >
                  English
                  {language === "en" && (
                    <motion.div
                      layoutId="active-language"
                      className="absolute inset-0 -z-10 rounded-full bg-gradient-to-r from-teal-500 to-cyan-600 shadow-md"
                      transition={{ type: "spring", stiffness: 300, damping: 30 }}
                    />
                  )}
                </span>
                <span
                  className={`relative z-10 rounded-full px-4 py-1.5 text-sm font-medium transition-colors duration-300 ${language === "si" ? "text-white" : "text-gray-500"
                    }`}
                >
                  සිංහල
                  {language === "si" && (
                    <motion.div
                      layoutId="active-language"
                      className="absolute inset-0 -z-10 rounded-full bg-gradient-to-r from-teal-500 to-cyan-600 shadow-md"
                      transition={{ type: "spring", stiffness: 300, damping: 30 }}
                    />
                  )}
                </span>
              </button>
            </div>

            <div className="mx-auto max-w-2xl">
              <div className="glow-input-wrapper">
                <ModernSimpleInput
                  value={value}
                  onChange={(e) => setValue(e.target.value)}
                  onClick={handleStartChat}
                  onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleStartChat()}
                  placeholder="Start a conversation..."
                  className="w-full !bg-white text-[12px] shadow-2xl transition-all duration-300 hover:shadow-teal-500/20 focus:shadow-teal-500/30 md:text-lg"
                />
              </div>
            </div>

            <div className="flex flex-wrap justify-center gap-x-10 gap-y-4 pt-3 font-mono text-sm text-gray-600">
              <span>Instant Responses</span>
              <span>Private & Secure</span>
              <span>Creative & Logical</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Landing;
