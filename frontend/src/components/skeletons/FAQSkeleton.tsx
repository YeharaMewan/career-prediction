import { motion } from "framer-motion";
import Skeleton from "../Skeleton";

const FAQSkeleton = () => {
  return (
    <motion.section
      className="relative w-full overflow-hidden bg-gradient-to-br from-teal-50 via-cyan-50 to-teal-100 py-24 lg:py-32"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
      role="status"
      aria-label="Loading FAQ content"
    >
      <span className="sr-only">Loading...</span>

      {/* Background orbs */}
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute top-20 left-10 h-80 w-80 rounded-full bg-teal-400/20 blur-3xl" />
        <div className="absolute right-20 bottom-32 h-96 w-96 rounded-full bg-cyan-500/20 blur-3xl" />
      </div>

      <div className="relative z-10 mx-auto max-w-7xl px-6">
        {/* Title skeleton */}
        <div className="text-center">
          <Skeleton className="mx-auto h-20 w-64 md:h-28 md:w-96" />
          <Skeleton className="mx-auto mt-6 h-6 w-3/4 max-w-3xl md:h-8" />
          <Skeleton className="mx-auto mt-2 h-6 w-2/3 max-w-2xl md:h-8" />
        </div>

        {/* FAQ cards skeleton */}
        <div className="mt-20 space-y-8">
          {[1, 2, 3, 4, 5, 6].map((item) => (
            <div
              key={item}
              className="overflow-hidden rounded-3xl border border-transparent bg-white/75 p-8 shadow-2xl backdrop-blur-2xl lg:p-10"
            >
              <div className="flex items-center justify-between">
                <Skeleton className="h-6 w-3/4 lg:h-8" />
                <Skeleton variant="circular" width="56px" height="56px" />
              </div>
            </div>
          ))}
        </div>

        {/* CTA skeleton */}
        <div className="mt-24 text-center">
          <Skeleton className="mx-auto h-8 w-96" />
          <Skeleton className="mx-auto mt-6 h-14 w-64 rounded-full" />
        </div>
      </div>
    </motion.section>
  );
};

export default FAQSkeleton;
