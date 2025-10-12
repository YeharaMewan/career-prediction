// import React, { useState, useRef } from "react";
// import Logo from "../assets/images/navbar/logo.png";

// //import Profile from "../assets/images/profile.webp";
// import useClickOutside from "../hooks/UseClickOutside";
// import { AlignJustify, X } from "lucide-react";

// interface NavbarProps {
//   sectionRefs?: { [key: string]: React.RefObject<HTMLDivElement | null> };
// }

// const Navbar: React.FC<NavbarProps> = ({ sectionRefs }) => {
//   const [isOpen, setIsOpen] = useState(false);
//   const navbarRef = useRef<HTMLDivElement>(null);

//   const toggleMenu = () => setIsOpen(!isOpen);

//   useClickOutside(navbarRef, () => {
//     if (isOpen) setIsOpen(false);
//   });

//   const handleScroll = (key: string) => {
//     if (!sectionRefs) return; // exit if undefined
//     sectionRefs[key]?.current?.scrollIntoView({ behavior: "smooth" });
//     setIsOpen(false); // close mobile menu after click
//   };

//   const navItems = [
//     { label: "Home", key: "Landing" },
//     { label: "How It Works", key: "Aboutus" },
//     { label: "Careers", key: "Technology" },
//     { label: "Testimonials", key: "Howaxis" },
//     { label: "Prediction", key: "Prediction" },
//     { label: "Contact", key: "RoadMap" },
//     // { label: "Blog", key: "FAQ" },
//     // { label: "FAQ", key: "FAQ" },
//   ];

//   return (
//     <nav className="start-0 top-0 z-20 w-full">
//       <div className="flex flex-wrap items-center justify-between p-4">
//         <div className="flex items-center gap-x-2">
//           <img
//             src={Logo}
//             onClick={() => handleScroll("Landing")}
//             className="h-auto w-full max-w-[80px] cursor-pointer"
//             alt="logo"
//           />
//           <span
//             onClick={() => handleScroll("Landing")}
//             className="font-fredoka cursor-pointer text-xl font-bold text-[#fc33a4]"
//           >
//             FutureMe
//           </span>
//         </div>

//         <div className="flex gap-[8px] md:order-2">
//           {/* <button className="font-fredoka relative flex items-center rounded-[16px] border-[1px] border-[#fc33a4] px-[12px] py-[8px] text-[16px] leading-[24px] text-[#fc33a4] uppercase">
//             {/* <a href="#">
//               <Search className="h-auto w-full cursor-pointer rounded-full border-[1px] border-white p-[5px] hover:bg-zinc-50" />
//             </a> */}
//             Start Prediction
//           {/* </button> */}
//           {/* <li className="relative flex items-center">
//             <a href="#">
//               <img
//                 src={Profile}
//                 alt="profile"
//                 className="h-9 w-9 cursor-pointer rounded-full bg-black object-contain p-2 hover:bg-black/80"
//               />
//             </a>
//           </li> */}

//           <button
//             onClick={toggleMenu}
//             type="button"
//             className="relative z-[60] inline-flex h-10 w-10 items-center justify-center rounded-full bg-[#000000] p-2 text-white lg:hidden"
//             aria-controls="navbar-sticky"
//             aria-expanded={isOpen ? "true" : "false"}
//           >
//             <span className="sr-only">Open main menu</span>
//             {isOpen ? <X className="h-4 w-4" /> : <AlignJustify className="h-5 w-5" />}
//           </button>
//         </div>

//         {/* Overlay for blur effect */}
//         {isOpen && <div className="fixed inset-0 z-40 bg-black/20 backdrop-blur-md lg:hidden" />}

//         <div
//           ref={navbarRef}
//           className={`lg:backdrop-blur-0 absolute top-0 right-0 z-50 w-[300px] rounded-tl-[24px] rounded-b-[24px] bg-black backdrop-blur-lg lg:relative lg:bg-transparent lg:bg-none ${
//             isOpen ? "flex" : "hidden lg:flex lg:w-auto lg:justify-center"
//           }`}
//         >
//           <div
//             className="w-full lg:flex lg:w-auto lg:items-center lg:justify-between"
//             id="navbar-sticky"
//           >
//             <ul
//               className={`font-fredoka flex flex-col space-y-4 p-4 pt-[78px] lg:mt-0 lg:flex-row lg:space-y-0 lg:space-x-4 lg:p-0 ${
//                 isOpen ? "" : ""
//               }`}
//             >
//               {navItems.map((item) => (
//                 <li key={item.label}>
//                   <button
//                     onClick={() => item.key && handleScroll(item.key)}
//                     className="font-fredoka inline-block w-[244px] cursor-pointer rounded-[16px] border border-[#fc33a4] py-4 text-center text-[16px] leading-[24px] font-bold text-white hover:border-b hover:border-[#fc33a4] hover:bg-[#fc33a4] lg:w-auto lg:rounded-none lg:border-0 lg:px-0 lg:py-0 lg:font-normal lg:hover:border-b lg:hover:bg-transparent"
//                   >
//                     {item.label}
//                   </button>
//                 </li>
//               ))}
//             </ul>
//           </div>
//         </div>
//       </div>
//     </nav>
//   );
// };

// export default Navbar;
import React from "react";
import Logo from "../assets/images/navbar/logo.png";

const Navbar: React.FC = () => {
  return (
    <nav className="start-0 top-0 z-20 w-full">
      <div className="flex items-center">
        <img src={Logo} alt="logo" className="h-auto w-full max-w-[80px]" />
      </div>
    </nav>
  );
};

export default Navbar;
