import React from "react";

interface LanguageToggleProps {
  currentLanguage: "en" | "si";
  onLanguageChange: (lang: "en" | "si") => void;
}

const LanguageToggle: React.FC<LanguageToggleProps> = ({
  currentLanguage,
  onLanguageChange,
}) => {
  return (
    <div className="fixed right-4 top-4 z-50">
      <div className="flex items-center gap-2 rounded-full border border-neutral-400/20 bg-neutral-800 p-1">
        <button
          onClick={() => onLanguageChange("en")}
          className={`rounded-full px-4 py-2 text-sm font-medium transition-all duration-200 ${
            currentLanguage === "en"
              ? "bg-sky-400/20 border border-sky-400/40 text-sky-400"
              : "text-neutral-400 hover:text-neutral-300"
          }`}
          aria-label="Switch to English"
        >
          EN
        </button>
        <button
          onClick={() => onLanguageChange("si")}
          className={`rounded-full px-4 py-2 text-sm font-medium transition-all duration-200 ${
            currentLanguage === "si"
              ? "bg-sky-400/20 border border-sky-400/40 text-sky-400"
              : "text-neutral-400 hover:text-neutral-300"
          }`}
          aria-label="Switch to Sinhala"
        >
          සිං
        </button>
      </div>
    </div>
  );
};

export default LanguageToggle;
