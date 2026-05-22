import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export const metadata: Metadata = {
  title: "Manas Mitra - AI Mental Health Companion",
  description: "Your compassionate AI companion for mental health support. Get immediate help, guidance, and professional referrals when you need them most.",
  keywords: ["mental health", "AI chatbot", "therapy", "counseling", "anxiety", "depression", "wellness"],
  authors: [{ name: "Manas Mitra Team" }],
  robots: "index, follow",
  openGraph: {
    title: "Manas Mitra - AI Mental Health Companion",
    description: "Your compassionate AI companion for mental health support",
    type: "website",
    locale: "en_US",
  },
};

import { ThemeProvider } from "@/components/theme-provider";

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning className={inter.variable}>
      <body className="font-sans antialiased">
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
