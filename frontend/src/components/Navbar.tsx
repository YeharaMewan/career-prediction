import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import Logo from "../assets/images/navbar/logo.png";
import { AlignJustify, X } from "lucide-react";

const Navbar: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const navigate = useNavigate();

  const goToFAQ = () => {
    navigate("/faq");
    setIsOpen(false);
    window.scrollTo(0, 0);
  };

  const toggleMenu = () => setIsOpen(prev => !prev);

  return (
    <>
      <nav className="fixed top-0 left-0 right-0 z-50 w-full border-b border-white/10 bg-black/70 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <Link to="/" className="flex items-center gap-3">
            <img src={Logo} alt="Career.AI" className="h-12 w-auto" />
            <span className="hidden font-mono text-4xl font-bold text-white hover:text-white/50 uppercase sm:block">
              Career.AI
            </span>
          </Link>

          <div className="hidden items-center gap-10 lg:flex">
            <button
              onClick={goToFAQ}
              className="font-mono text-2xl font-medium text-white transition hover:text-white/50"
            >
              FAQ
            </button>
          </div>

          <button
            onClick={toggleMenu}
            className="relative z-50 lg:hidden"
            aria-label="Toggle menu"
          >
            {isOpen ? (
              <X className="h-8 w-8 text-white" />
            ) : (
              <AlignJustify className="h-8 w-8 text-white" />
            )}
          </button>
        </div>
      </nav>


      {isOpen && (
        <div className="fixed inset-0 z-40 flex lg:hidden">

          <div
            className="absolute inset-0 bg-black/80 backdrop-blur-sm"
            onClick={() => setIsOpen(false)}
          />

          <div className="relative ml-auto w-full max-w-sm bg-gradient-to-b from-black via-purple-900/50 to-black p-8 shadow-2xl">
            <div className="mt-24 space-y-6">
              <button
                onClick={goToFAQ}
                className="block w-full rounded-xl bg-white/10 py-6 text-center font-mono text-3xl font-bold text-white backdrop-blur-md transition hover:bg-pink-600/40"
              >
                FAQ
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default Navbar;