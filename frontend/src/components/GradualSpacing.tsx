"use client";

import { motion, useInView } from "framer-motion";
import * as React from "react";

export function GradualSpacing({ text, className = "" }: { text: string; className?: string }) {
  const ref = React.useRef(null);
  const isInView = useInView(ref, { once: false });

  return (
    <div ref={ref} className={className} style={{ display: "inline" }}>
      {text.split("").map((char, i) => (
        <motion.span
          key={i}
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : {}}
          transition={{
            duration: 0.4,
            delay: i * 0.03, // slightly faster for smoother text
          }}
          style={{ display: "inline-block" }} // animate without breaking words
        >
          {char === " " ? "\u00A0" : char}
        </motion.span>
      ))}
    </div>
  );
}
