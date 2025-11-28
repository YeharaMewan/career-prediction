import { useState, useEffect } from "react";
import FAQSkeleton from "../../../components/skeletons/FAQSkeleton";

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
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Simulate content loading
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 600);

    return () => clearTimeout(timer);
  }, []);

  if (isLoading) {
    return <FAQSkeleton />;
  }

  return (
    <section className="relative w-full overflow-hidden bg-gradient-to-br from-teal-50 via-cyan-50 to-teal-100 pt-20 pb-20">
      {/* Cool Animated Background Orbs */}
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute top-20 left-10 h-80 w-80 animate-pulse rounded-full bg-teal-400/30 blur-3xl" />
        <div className="absolute right-20 bottom-32 h-96 w-96 animate-pulse rounded-full bg-cyan-500/30 blur-3xl delay-1000" />
        <div className="absolute top-1/2 left-1/2 h-96 w-96 -translate-x-1/2 -translate-y-1/2 rounded-full bg-teal-300/20 blur-3xl" />
      </div>

      <div className="relative z-10 mx-auto max-w-7xl px-6">
        {/* Title */}
        <div className="text-center">
          <h2 className="inline-flex items-center gap-4 text-5xl font-bold text-teal-700 md:text-7xl lg:text-7xl">
            Frequently Asked Question
          </h2>
          <p className="mx-auto mt-6 max-w-3xl font-mono text-xl font-light text-gray-600 md:text-2xl">
            Everything you wanted to know about shaping your future with AI
          </p>
        </div>

        {/* FAQ Cards – Warm Gold/Amber Theme */}
        <div className="mt-20 space-y-8">
          {faqData.map((faq, index) => (
            <div
              key={index}
              className="group overflow-hidden rounded-3xl border border-transparent bg-white/75 shadow-2xl backdrop-blur-2xl transition-all duration-500 hover:border-teal-400/60 hover:bg-white/95 hover:shadow-[0_0_60px_rgba(56,163,165,0.25)]"
            >
              {/* Question + Icon */}
              <div className="flex items-center justify-between pt-8 pl-8 pb-3 lg:pt-8 lg:pb-3 lg:pl-10">
                <h3 className="max-w-4xl pr-8 font-mono text-lg font-medium text-gray-900 lg:text-2xl">
                  {faq.question}
                </h3>
              </div>

              {/* Answer – smooth expand */}
              <div
                className={`grid transition-all duration-700 ease-in-out `}
              >
                <div className="overflow-hidden px-8 pb-10 lg:px-10">
                  <p className="font-mono text-base leading-relaxed text-gray-700 lg:text-lg">
                    {faq.answer}
                  </p>
                </div>
              </div>

              {/* Glowing bottom bar – cool gradient */}
              <div
                className={`h-1 bg-gradient-to-r from-teal-500 via-cyan-500 to-teal-400 transition-all duration-700`}
              />
            </div>
          ))}
        </div>

        {/* Final CTA – matching cool theme */}
        <div className="mt-24 text-center">
          <p className="text-2xl font-light text-gray-700">Ready to meet your future self?</p>
          <button className="mt-6 rounded-full bg-gradient-to-r from-teal-500 to-cyan-600 px-12 py-5 text-[16px] font-bold text-white shadow-2xl transition-all hover:scale-105 hover:shadow-teal-600/60 lg:text-xl">
            Start Now – It's Free
          </button>
        </div>
      </div>
    </section>
  );
};

export default FAQSection;
