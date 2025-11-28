import { useState, useRef } from "react";
import useClickOutside from "../../../hooks/UseClickOutside";
import PreviewUseAutoScroll from "../../../components/PreviewUseAutoScroll";
import { ModernSimpleInput } from "../../../components/Input";

const Landing = () => {
  const [chatOpen, setChatOpen] = useState(false);
  const [value, setValue] = useState("");
  const chatRef = useRef<HTMLDivElement>(null);

  useClickOutside(chatRef, () => setChatOpen(false));

  return (
    <div className="relative w-full">
      {chatOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-md">
          <div
            ref={chatRef}
            className="h-[90vh] w-[90vw] max-w-4xl overflow-hidden rounded-2xl bg-neutral-900 shadow-2xl"
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
      )}

      {!chatOpen && (
        <div className="relative flex w-full items-center justify-center px-6 py-20">
          <div className="pointer-events-none absolute inset-0 overflow-hidden">
            <div className="absolute -top-40 -right-40 h-90 w-96 rounded-full bg-orange-300/30 blur-3xl" />
            <div className="absolute -bottom-40 -left-40 h-90 w-96 rounded-full bg-amber-400/30 blur-3xl" />
          </div>

          <div className="relative z-10 w-full max-w-5xl text-center">
            <div className="space-y-7 xl:space-y-9">
              <h1 className="text-3xl leading-tight font-bold text-gray-900 md:text-6xl lg:text-7xl xl:text-8xl">
                Build Imagination
                <br />
                <span className="bg-gradient-to-r from-yellow-500 to-amber-500 bg-clip-text text-transparent">
                  With Logic AI
                </span>
              </h1>

              <p className="mx-auto max-w-2xl font-mono text-lg font-light text-gray-700 md:text-xl">
                Your intelligent companion for creativity, problem-solving, and turning ideas into
                reality.
              </p>

              <div className="mx-auto max-w-2xl">
                <ModernSimpleInput
                  value={value}
                  onChange={(e) => setValue(e.target.value)}
                  onClick={() => setChatOpen(true)}
                  onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && setChatOpen(true)}
                  placeholder="Start a conversation..."
                  className="w-full text-[12px] shadow-2xl transition-all duration-300 hover:shadow-violet-500/20 focus:shadow-violet-500/30 md:text-lg"
                />
              </div>

              <div className="flex flex-wrap justify-center gap-x-10 gap-y-4 pt-3 font-mono text-sm text-gray-600">
                <span>Instant Responses</span>
                <span>Private & Secure</span>
                <span>Creative & Logical</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Landing;
