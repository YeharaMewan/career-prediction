import { useState, useRef } from "react";
import PreviewUseAutoScroll from "../../../components/PreviewUseAutoScroll";
import useClickOutside from "../../../hooks/UseClickOutside";
import { ModernSimpleInput } from "../../../components/Input";
import AIninja from "../../../assets/images/home/AIrobo.jpg";

const Landing = () => {
  const [chatOpen, setChatOpen] = useState(false);
  const chatRef = useRef<HTMLDivElement>(null);
  const [value, setValue] = useState("");

  useClickOutside(chatRef, () => setChatOpen(false));

  return (
    <div className="relative z-[0] w-full">
      {chatOpen ? (
        <div className="fixed inset-0 z-[50] flex w-full items-center justify-center bg-black/70">
          <div
            ref={chatRef}
            className="flex h-[90%] w-[90%] max-w-3xl items-center justify-center overflow-hidden rounded-xl bg-neutral-900 p-[24px] shadow-lg"
          >
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
