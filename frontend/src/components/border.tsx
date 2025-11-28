
import React from "react";

interface BorderProps {
  children: React.ReactNode;
  borderRadius?: string;
}

const Border: React.FC<BorderProps> = ({ children, borderRadius = "1rem" }) => {
  return (
    <>
      <style>{`
        .animated-border::before {
          animation: spin-gradient 10s linear infinite;
        }

        @keyframes spin-gradient {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>

      <div
        className="animated-border relative p-[2px] rounded-xl overflow-hidden"
        style={{ borderRadius }}
      >
        <div
          className="absolute inset-0 bg-[conic-gradient(#00F5FF,#00F5FF30,#00F5FF)]"
          style={{ animation: "spin-gradient 10s linear infinite" }}
        />

        {/* FIXED */}
        <div className="relative bg-white rounded-xl p-5 overflow-hidden">
          {children}
        </div>
      </div>
    </>
  );
};

export default Border;
