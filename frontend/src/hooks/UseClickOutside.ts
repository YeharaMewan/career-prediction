import { useEffect, RefObject } from "react";

const useClickOutside = <T extends HTMLElement | null>(
  ref: RefObject<T>,
  callback: () => void,
  parentRef: RefObject<HTMLElement> | null = null,
) => {
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node;
      if (ref.current && !ref.current.contains(target)) {
        if (!parentRef || (parentRef.current && parentRef.current.contains(target))) {
          callback();
        }
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [ref, parentRef, callback]);
};

export default useClickOutside;
