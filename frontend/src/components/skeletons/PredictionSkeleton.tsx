import { motion } from "framer-motion";
import Skeleton from "../Skeleton";

const PredictionSkeleton = () => {
  return (
    <motion.div
      className="flex h-screen items-center justify-center"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
      role="status"
      aria-label="Loading prediction"
    >
      <span className="sr-only">Loading...</span>

      <div className="wrapper">
        <div className="relative flex flex-col items-center justify-center">
          {/* Central loading text */}
          <Skeleton className="mb-8 h-12 w-96" />

          {/* Rotating cards placeholder */}
          <div className="relative h-48 w-32">
            <Skeleton
              className="absolute inset-0 rounded-xl"
              animate={true}
            />
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default PredictionSkeleton;
