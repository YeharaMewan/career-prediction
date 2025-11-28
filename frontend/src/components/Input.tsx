// import { type InputHTMLAttributes, forwardRef } from "react";
// import { cn } from "../utils/cn";

// export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
//   label?: string;
// }

// const ModernSimpleInput = forwardRef<HTMLInputElement, InputProps>(
//   ({ className, type, ...props }, ref) => {
//     return (
//       <input
//         className={cn(
//           "flex h-30 w-full rounded-xl border px-3 py-2  text-sm ring-0 transition focus:ring-0 focus-visible:outline-hidden disabled:cursor-not-allowed disabled:opacity-50",

//           "bg-[#E6D8A8]",
//           "placeholder:text-neutral-400",
//           // File
//           "font-fredoka file:text-foreground file:border-0 file:bg-transparent file:text-sm file:font-medium",
//           className,
//         )}
//         ref={ref}
//         type={type}
//         {...props}
//       />
//     );
//   },
// );
// ModernSimpleInput.displayName = "Modern Simple Input";

// export { ModernSimpleInput };
import { type InputHTMLAttributes, forwardRef } from "react";
import { cn } from "../utils/cn";

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: boolean;
}

const ModernSimpleInput = forwardRef<HTMLInputElement, InputProps>(
  ({ className, type = "text", label, error, disabled, ...props }, ref) => {
    return (
      <div className="relative w-full">
        {label && <label className="mb-2 block text-sm font-medium text-gray-700">{label}</label>}

        <input
          type={type}
          ref={ref}
          disabled={disabled}
          className={cn(
            // Base Layout & Size
            "w-full rounded-2xl border-2 px-6 py-5 text-base transition-all duration-300",
            "placeholder:font-light placeholder:text-gray-400",
            "font-medium tracking-wide",

            // Background & Border
            "bg-white/80 backdrop-blur-xl",
            "border-transparent",

            // Focus States (Beautiful Glow)
            "focus:border-teal-400 focus:ring-4 focus:ring-teal-500/20",
            "focus:shadow-xl focus:shadow-teal-500/10 focus:outline-none",

            // Hover
            "hover:border-teal-300 hover:shadow-lg hover:shadow-teal-500/5",

            // States
            "disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:border-transparent",
            error && "border-red-400 focus:border-red-400 focus:ring-red-500/20",

            // Glassmorphism Inner Shadow
            "shadow-inner shadow-white/50",

            // File input styling (if ever used)
            "file:cursor-pointer file:rounded-lg file:border-0 file:bg-teal-600 file:px-4 file:py-2 file:font-medium file:text-white",

            className,
          )}
          {...props}
        />

        {/* Optional: Floating Send Icon (only show if inside chat input) */}
        {props.placeholder?.includes("message") || props.placeholder?.includes("Ask") ? (
          <div className="pointer-events-none absolute inset-y-0 right-4 flex items-center">
            <div className="rounded-lg bg-gradient-to-r from-teal-600 to-cyan-700 p-2.5 text-white opacity-0 transition-all duration-300 group-focus-within:opacity-100">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-5 w-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                />
              </svg>
            </div>
          </div>
        ) : null}
      </div>
    );
  },
);

ModernSimpleInput.displayName = "ModernSimpleInput";

export { ModernSimpleInput };
