import { useState } from "react";
import { ChevronDown, ChevronUp, Sparkles } from "lucide-react";

const faqData = [
  {
    question: "What makes FutureMe unique?",
    answer:
      "We don’t just answer — we help you visualize and build the future version of yourself using emotional intelligence, creativity, and logic.",
  },
  {
    question: "Is it really free to use?",
    answer: "Yes! Start instantly — no signup, no credit card. Premium features coming soon.",
  },
  {
    question: "How accurate are the predictions?",
    answer:
      "Our AI combines psychology, trend analysis, and your personal inputs to create inspiring, realistic future scenarios.",
  },
  {
    question: "Can I use it on my phone?",
    answer: "100%. FutureMe is fully mobile-optimized and feels native on any device.",
  },
  {
    question: "Is my data safe?",
    answer: "Absolutely. All conversations are encrypted and never stored or shared.",
  },
  {
    question: "What’s next for FutureMe?",
    answer:
      "Vision boards, voice mode, career path simulations, and collaborative future-building with friends.",
  },
];

const FAQSection = () => {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  const toggleAnswer = (index: number) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <section className="relative w-full overflow-hidden bg-gradient-to-br from-amber-50 via-orange-50 to-amber-100 py-24 lg:py-32">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute top-20 left-10 h-80 w-80 animate-pulse rounded-full bg-orange-400/30 blur-3xl" />
        <div className="absolute right-20 bottom-32 h-96 w-96 animate-pulse rounded-full bg-amber-500/30 blur-3xl delay-1000" />
        <div className="absolute top-1/2 left-1/2 h-96 w-96 -translate-x-1/2 -translate-y-1/2 rounded-full bg-yellow-400/20 blur-3xl" />
      </div>

      <div className="relative z-10 mx-auto max-w-7xl px-6">
        <div className="text-center">
          <h2 className="inline-flex items-center gap-4 text-5xl font-bold text-gray-900 md:text-7xl lg:text-8xl">
            FAQ
            <Sparkles className="h-12 w-12 text-orange-500 md:h-16 md:w-16" />
          </h2>
          <p className="mx-auto mt-6 max-w-3xl font-mono text-xl font-light text-gray-700 md:text-2xl">
            Everything you wanted to know about shaping your future with AI
          </p>
        </div>
        <div className="mt-20 space-y-8">
          {faqData.map((faq, index) => (
            <div
              key={index}
              className="group overflow-hidden rounded-3xl border border-transparent bg-white/75 shadow-2xl backdrop-blur-2xl transition-all duration-500 hover:border-orange-400/60 hover:bg-white/95 hover:shadow-[0_0_60px_rgba(251,146,60,0.25)]"
            >
              <div className="flex items-center justify-between p-8 lg:p-10">
                <h3 className="max-w-4xl pr-8 font-mono text-lg font-medium text-gray-900 lg:text-2xl">
                  {faq.question}
                </h3>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleAnswer(index);
                  }}
                  className="flex-shrink-0 rounded-full bg-gradient-to-r from-orange-500 to-amber-500 p-4 text-white shadow-lg ring-4 ring-orange-300/30 transition-all duration-300 hover:scale-110 hover:shadow-2xl hover:shadow-orange-500/40 focus:ring-8 focus:ring-orange-400/30 focus:outline-none"
                  aria-label={openIndex === index ? "Close" : "Open"}
                >
                  {openIndex === index ? (
                    <ChevronUp className="h-6 w-6" />
                  ) : (
                    <ChevronDown className="h-6 w-6" />
                  )}
                </button>
              </div>

              <div
                className={`grid transition-all duration-700 ease-in-out ${
                  openIndex === index ? "grid-rows-[1fr]" : "grid-rows-[0fr]"
                }`}
              >
                <div className="overflow-hidden px-8 pb-10 lg:px-10">
                  <p className="font-mono text-base leading-relaxed text-gray-700 lg:text-lg">
                    {faq.answer}
                  </p>
                </div>
              </div>

              <div
                className={`h-1.5 bg-gradient-to-r from-orange-500 via-amber-500 to-yellow-500 transition-all duration-700 ${
                  openIndex === index ? "opacity-100" : "opacity-0"
                }`}
              />
            </div>
          ))}
        </div>
        <div className="mt-24 text-center">
          <p className="text-2xl font-light text-gray-700">Ready to meet your future self?</p>
          <button className="mt-6 rounded-full bg-gradient-to-r from-orange-500 to-amber-500 px-12 py-5 text-[16px] font-bold text-white shadow-2xl transition-all hover:scale-105 hover:shadow-orange-600/60 lg:text-xl">
            Start Now – It’s Free
          </button>
        </div>
      </div>
    </section>
  );
};

export default FAQSection;
