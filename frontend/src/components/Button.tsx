import React from "react";

interface ButtonProps {
  label: string;
  onClick?: () => void;
  className?: string; 
  bg?: string;
}

const Button: React.FC<ButtonProps> = ({
  label,
  onClick,
  bg,
  className = "",
}) => {
  return (
    <>
      {/* <style>{`
        .button-wrapper::before {
          animation: spin-gradient 4s linear infinite;
        }

        @keyframes spin-gradient {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style> */}

      <div className="button-wrapper relative inline-block p-0.5 rounded-full overflow-hidden hover:scale-105 transition duration-300 active:scale-100 before:content-[''] before:absolute before:inset-0 ">
        <button
          onClick={onClick}
          className={`relative z-10 ${bg} text-white rounded-full px-8 py-3 font-medium text-sm ${className}`}
        >
          {label}
        </button>
      </div>
    </>
  );
};

export default Button;
