import { useState, useRef } from "react";
import PreviewUseAutoScroll from "../../../components/PreviewUseAutoScroll";
import { ModernSimpleInput } from "../../../components/Input";
import LanguageToggle from "../../../components/LanguageToggle";
import AIninja from "../../../assets/images/home/AIrobo.jpg";

const Landing = () => {
  const [chatOpen, setChatOpen] = useState(false);
  const chatRef = useRef<HTMLDivElement>(null);
  const [value, setValue] = useState("");

  // Language state with localStorage persistence (syncs with chat)
  const [language, setLanguage] = useState<"en" | "si">(() => {
    return (localStorage.getItem("preferredLanguage") as "en" | "si") || "en";
  });

  // Handle language change
  const handleLanguageChange = (newLang: "en" | "si") => {
    setLanguage(newLang);
    localStorage.setItem("preferredLanguage", newLang);
  };

  // Background click navigation disabled - close icon in chat interface will handle closing

  return (
    <div className="relative z-[0] w-full">
      {chatOpen ? (
        <div className="fixed inset-0 z-[50] flex w-full items-center justify-center bg-black/70">
          <div
            ref={chatRef}
            className="relative flex h-[90%] w-[90%] max-w-3xl items-center justify-center overflow-hidden rounded-xl bg-neutral-900 p-[24px] shadow-lg"
          >
            {/* Close button */}
            <button
              onClick={() => setChatOpen(false)}
              className="absolute top-6 right-6 z-10 rounded-lg p-2 text-neutral-400 hover:bg-neutral-700/50 hover:text-neutral-200 transition-colors"
              aria-label="Close chat"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
            <PreviewUseAutoScroll />
          </div>
        </div>
      ) : (
        <div className="flex w-full flex-col items-center justify-between lg:flex-row lg:gap-x-[50px]">
          <div className="flex w-full flex-col justify-center gap-y-[24px]">
            <div className="w-full items-center text-center">
              <p className="font-fredoka pb-[16px] text-[52px] leading-[72px] font-bold text-white uppercase xl:text-[86px] xl:leading-[90px] 2xl:leading-[140px]">
                Build Imagination on your mind with logic ai
              </p>
            </div>

            {/* Language selector - syncs with chat via localStorage */}
            <div className="flex w-full justify-center mb-4">
              <LanguageToggle
                currentLanguage={language}
                onLanguageChange={handleLanguageChange}
              />
            </div>

            <div className="codepen-button z-50">
              <ModernSimpleInput
                onClick={() => setChatOpen(true)}
                onChange={(e) => setValue(e.target.value)}
                placeholder="Say Hi..."
                type="text"
                value={value}
                className=" relative z-50"
              />
            </div>
          </div>

          <div className="z-[50] flex w-full justify-end">
            <img src={AIninja} alt="" className="z-[2] h-[88vh] rounded-[24px] object-cover" />
          </div>
        </div>
      )}
    </div>
  );
};

export default Landing;
