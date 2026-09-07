import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "PneumoScan AI — Explainable Chest X-Ray Pneumonia Detection" },
      {
        name: "description",
        content:
          "Upload a chest X-ray and get an AI pneumonia prediction with Grad-CAM and SHAP explainability.",
      },
      { property: "og:title", content: "PneumoScan AI" },
      {
        property: "og:description",
        content:
          "Futuristic AI chest X-ray pneumonia detection with Grad-CAM and SHAP explainability.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: Index,
});

function Index() {
  return (
    <iframe
      src="/preview.html"
      title="PneumoScan AI"
      className="h-screen w-full border-0"
    />
  );
}
