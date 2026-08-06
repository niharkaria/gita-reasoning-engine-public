import type { Metadata } from "next";
import { Fraunces, Work_Sans, Noto_Sans_Gujarati } from "next/font/google";
import "./globals.css";

const fraunces = Fraunces({
  subsets: ["latin"],
  variable: "--font-display",
  weight: ["400", "500", "600"],
});

const workSans = Work_Sans({
  subsets: ["latin"],
  variable: "--font-body",
  weight: ["400", "500", "600"],
});

const notoSansGujarati = Noto_Sans_Gujarati({
  subsets: ["gujarati"],
  variable: "--font-gujarati",
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "Gita Reasoning Engine",
  description: "Retrieval-grounded Q&A over a defined Pushtimarg Bhagavad Gita corpus.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${fraunces.variable} ${workSans.variable} ${notoSansGujarati.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-stone">{children}</body>
    </html>
  );
}
