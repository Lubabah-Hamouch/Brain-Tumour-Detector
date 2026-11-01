import type { Metadata } from "next";
import { Geist, JetBrains_Mono } from "next/font/google";

import { SiteFooter } from "@/components/SiteFooter";
import { SiteHeader } from "@/components/SiteHeader";
import { cssVariables } from "@/design/tokens";
import "./globals.css";

// next/font downloads both families at build time and serves them from this app:
// the browser never contacts Google.
const sans = Geist({ variable: "--font-sans", subsets: ["latin"] });
const mono = JetBrains_Mono({ variable: "--font-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: { default: "Brain Tumour MRI", template: "%s · Brain Tumour MRI" },
  description:
    "Research demonstration: 3D tumour segmentation and tumour-type classification for brain MRI, evaluated on held-out test sets. Not a medical device.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <head>
        <style dangerouslySetInnerHTML={{ __html: cssVariables() }} />
      </head>
      <body>
        <SiteHeader />
        {children}
        <SiteFooter />
      </body>
    </html>
  );
}
