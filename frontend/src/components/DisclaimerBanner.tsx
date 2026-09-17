export function DisclaimerBanner({ text }: { text: string }) {
  return (
    <div className="w-full bg-amber-100 border-b border-amber-300 text-amber-900 text-sm px-4 py-2 text-center">
      ⚠️ {text}
    </div>
  );
}
