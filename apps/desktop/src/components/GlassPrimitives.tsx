import type { ButtonHTMLAttributes, ReactNode } from "react";

export function Glyph({ children }: { children: ReactNode }) {
  return <span className="glyph" aria-hidden="true">{children}</span>;
}

export function GlassButton({ children, className = "", ...props }: ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button className={`glass-button ${className}`} {...props}>{children}</button>;
}
