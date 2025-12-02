
import React from "react";

interface BorderProps {
  children: React.ReactNode;
  borderRadius?: string;
  color?: string;
}

const Border: React.FC<BorderProps> = ({ children, borderRadius = "1rem", color }) => {
  const backgroundGradient = color
    ? `conic-gradient(from 0deg, ${color}, ${color}30, ${color})`
    : `conic-gradient(from 0deg, #abffb3, #80ed99, #57cc99, #38a3a5, #57cc99, #80ed99, #abffb3)`;

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
          className="absolute inset-0"
          style={{
            background: backgroundGradient,
            animation: "spin-gradient 15s linear infinite"
          }}
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
