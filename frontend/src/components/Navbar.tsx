// components/Navbar.tsx
import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { AlignJustify, X } from "lucide-react";

const Navbar: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const navigate = useNavigate();

  const goToFAQ = () => {
    navigate("/faq");
    setIsOpen(false);
    window.scrollTo(0, 0);
  };

  return (
    <nav className="fixed top-0 z-50 w-full border-b border-gray-200/30 bg-white/50 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
        <Link to="/" className="flex items-center">
          <span className="bg-gradient-to-r from-teal-400 via-teal-500 to-green-500 bg-clip-text text-3xl font-bold uppercase text-transparent sm:text-3xl lg:text-4xl transition-all duration-300 hover:scale-105 hover:from-green-500 hover:via-teal-500 hover:to-teal-500">
            Career.AI
          </span>
        </Link>

        <div className="hidden items-center gap-10 lg:flex">
          <button
            onClick={goToFAQ}
            className="relative text-lg font-semibold text-gray-800 transition duration-300 hover:text-transparent hover:bg-gradient-to-r hover:from-teal-500 hover:to-green-500 hover:bg-clip-text"
          >
            FAQ
          </button>
        </div>

        {/* CTA */}
        {/* <div className="hidden lg:block">
          <Link
            to="/prediction"
            className="rounded-full bg-gradient-to-r from-pink-500 to-purple-600 px-8 py-3 font-bold text-white shadow-lg transition hover:scale-105"
          >
            Start Prediction
          </Link>
        </div> */}

        {/* Mobile Toggle */}
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="relative rounded-lg p-2 transition duration-300 hover:bg-gray-300/20 active:scale-95 lg:hidden"
        >
          {isOpen ? (
            <X className="h-7 w-7 text-gray-800 transition duration-300" />
          ) : (
            <AlignJustify className="h-7 w-7 text-gray-800 transition duration-300" />
          )}
        </button>
      </div>

      {/* Mobile Menu */}
      {isOpen && (
        <>
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm lg:hidden" onClick={() => setIsOpen(false)} />
          <div className="fixed top-0 right-0 h-full w-80 bg-white/70 backdrop-blur-lg p-8 shadow-2xl shadow-black/10 border-l border-gray-200/30">
            <div className="mt-24 space-y-6">
              <button
                onClick={goToFAQ}
                className="block w-full cursor-pointer rounded-lg bg-teal-500/10 py-5 text-xl font-semibold text-gray-800 transition duration-300 hover:bg-gradient-to-r hover:from-teal-500/30 hover:to-green-500/20 active:scale-95"
              >
                FAQ
              </button>
              {/* <Link
                to="/prediction"
                onClick={() => setIsOpen(false)}
                className="block w-full rounded-lg bg-gradient-to-r from-pink-500 to-purple-600 py-5 text-center text-2xl font-bold text-white"
              >
                Start Prediction
              </Link> */}
            </div>
          </div>
        </>
      )}
    </nav>
  );
};

export default Navbar;
