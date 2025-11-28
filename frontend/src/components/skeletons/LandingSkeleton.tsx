import { motion } from "framer-motion";
import Skeleton from "../Skeleton";

const LandingSkeleton = () => {
  return (
    <motion.div
      className="relative flex w-full items-center justify-center px-6 py-20"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
      role="status"
      aria-label="Loading content"
    >
      <span className="sr-only">Loading...</span>

      {/* Background orbs - static for skeleton */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 h-90 w-96 rounded-full bg-teal-300/20 blur-3xl" />
        <div className="absolute -bottom-40 -left-40 h-90 w-96 rounded-full bg-cyan-400/20 blur-3xl" />
      </div>

      <div className="relative z-10 w-full max-w-5xl text-center">
        <div className="space-y-7 xl:space-y-9">
          {/* Main heading skeleton */}
          <div className="mx-auto space-y-4">
            <Skeleton className="mx-auto h-16 w-3/4 md:h-24 lg:h-28" />
            <Skeleton className="mx-auto h-16 w-2/3 md:h-24 lg:h-28" />
          </div>

          {/* Subtitle skeleton */}
          <div className="mx-auto max-w-2xl space-y-2">
            <Skeleton className="mx-auto h-6 w-5/6 md:h-7" />
            <Skeleton className="mx-auto h-6 w-4/6 md:h-7" />
          </div>

          {/* Input skeleton */}
          <div className="mx-auto max-w-2xl">
            <Skeleton className="h-14 w-full rounded-xl md:h-16" />
          </div>

          {/* Feature tags skeleton */}
          <div className="flex flex-wrap justify-center gap-x-10 gap-y-4 pt-3">
            <Skeleton className="h-5 w-32" />
            <Skeleton className="h-5 w-32" />
            <Skeleton className="h-5 w-32" />
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default LandingSkeleton;
