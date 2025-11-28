import { motion } from "framer-motion";

interface SkeletonProps {
  className?: string;
  variant?: "text" | "circular" | "rectangular";
  width?: string;
  height?: string;
  animate?: boolean;
}

const Skeleton = ({
  className = "",
  variant = "rectangular",
  width,
  height,
  animate = true
}: SkeletonProps) => {
  const baseClasses = "bg-gradient-to-r from-gray-200 via-gray-300 to-gray-200";

  const variantClasses = {
    text: "rounded h-4",
    circular: "rounded-full",
    rectangular: "rounded-lg"
  };

  const style = {
    width: width || "100%",
    height: height || (variant === "text" ? "1rem" : "100%"),
    backgroundSize: "200% 100%"
  };

  return (
    <motion.div
      className={`${baseClasses} ${variantClasses[variant]} ${className}`}
      style={style}
      animate={animate ? {
        backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"]
      } : undefined}
      transition={animate ? {
        duration: 1.5,
        repeat: Infinity,
        ease: "linear"
      } : undefined}
    />
  );
};

export default Skeleton;
