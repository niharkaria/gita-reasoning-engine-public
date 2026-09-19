import type { Metadata } from "next";
import { Playfair_Display, Work_Sans, Noto_Sans_Gujarati } from "next/font/google";
import "./globals.css";

// NOTE (2026-09-19): these `variable` names were renamed from
// --font-display / --font-body / --font-gujarati to --font-nf-* to
// avoid colliding with Tailwind v4's OWN token names of the same kind
// declared in globals.css's @theme block. Two different mechanisms
// (next/font on <html>, Tailwind's @theme at :root) both trying to own
// a custom property with the identical name is a real cascade footgun
// -- see globals.css's @theme comment for the full explanation.
const playfairDisplay = Playfair_Display({
  subsets: ["latin"],
  variable: "--font-nf-display",
  weight: ["400", "500", "600", "700", "800"],
});

const workSans = Work_Sans({
  subsets: ["latin"],
  variable: "--font-nf-body",
  weight: ["400", "500", "600"],
});

const notoSansGujarati = Noto_Sans_Gujarati({
  subsets: ["gujarati"],
  variable: "--font-nf-gujarati",
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "Gita Reasoning Engine",
  description: "Retrieval-grounded Q&A over a defined Pushtimarg Bhagavad Gita corpus.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${playfairDisplay.variable} ${workSans.variable} ${notoSansGujarati.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}