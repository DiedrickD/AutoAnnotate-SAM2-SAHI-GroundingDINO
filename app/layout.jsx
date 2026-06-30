import "./globals.css";

export const metadata = {
  title: "Grounded SAM 2 Studio",
  description: "Drag, prompt, and detect boxes or segment masks with GroundingDINO and optional SAM 2.",
};

export const viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
