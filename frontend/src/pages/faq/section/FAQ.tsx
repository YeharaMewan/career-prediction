import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";

const faqData = [
  {
    question: "What is FutureMe?",
    answer:
      "FutureMe is a utility token used for fast, secure, and low-cost digital payments. It also gives access to special services and rewards.",
  },
  {
    question: "What is FutureMe?",
    answer:
      "FutureMe is a utility token used for fast, secure, and low-cost digital payments. It also gives access to special services and rewards.",
  },
  {
    question: "What is FutureMe?",
    answer:
      "FutureMe is a utility token used for fast, secure, and low-cost digital payments. It also gives access to special services and rewards.",
  },
  {
    question: "What is FutureMe?",
    answer:
      "FutureMe is a utility token used for fast, secure, and low-cost digital payments. It also gives access to special services and rewards.",
  },
  {
    question: "What is FutureMe?",
    answer:
      "FutureMe is a utility token used for fast, secure, and low-cost digital payments. It also gives access to special services and rewards.",
  },
  {
    question: "What is FutureMe?",
    answer:
      "FutureMe is a utility token used for fast, secure, and low-cost digital payments. It also gives access to special services and rewards.",
  },
];

function FAQSection() {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  const toggle = (index: number) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <div className="relative rounded-[24px] px-[14px] py-[16px] lg:px-[63px] lg:py-[56px]">
      <section className="flex">
        <div className="w-full rounded-[24px] bg-[#161616]">
          <div className="rounded-[24px] p-[16px] lg:p-[36px]">
            <p className="font-fredoka pb-[16px] text-center text-[32px] leading-[40px] font-bold text-white lg:pb-[24px] lg:text-[64px] lg:leading-[72px]">
              Your Questions, Answeres
            </p>

            <div className="mx-auto grid grid-cols-1 gap-[24px] md:grid-cols-2">
              {faqData.map((faq, index) => (
                <div
                  key={index}
                  className="w-full cursor-pointer rounded-[24px] border-[1px] border-[#fc33a4] p-[16px] transition-shadow hover:shadow-md"
                  onClick={() => toggle(index)}
                >
                  <div className="flex items-center justify-between">
                    <p className="font-fredoka text-[18px] leading-[24px] font-bold text-white">
                      {faq.question}
                    </p>
                    {openIndex === index ? (
                      <ChevronUp className="h-5 w-5 text-gray-600" />
                    ) : (
                      <ChevronDown className="h-5 w-5 text-gray-600" />
                    )}
                  </div>
                  {openIndex === index && (
                    <p className="font-fredoka mt-2 text-[16px] leading-[24px] font-normal text-white">
                      {faq.answer}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

export default FAQSection;
