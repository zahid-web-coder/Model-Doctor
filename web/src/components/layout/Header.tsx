import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export function Header() {
  return (
    <header className="flex items-center justify-between h-14">
      {/* Left side: Selectors */}
      <div className="flex items-center gap-6">
        <div className="flex flex-col gap-1">
          <span className="text-[10px] text-slate font-medium uppercase tracking-wider">Model</span>
          <Select defaultValue="resnet50">
            <SelectTrigger className="h-6 p-0 border-0 shadow-none bg-transparent hover:bg-transparent focus:ring-0 text-[14px] font-medium text-ink w-auto gap-1 [&>svg]:opacity-50">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="resnet50">ResNet50</SelectItem>
              <SelectItem value="mobilenet">MobileNet V3</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="flex flex-col gap-1">
          <span className="text-[10px] text-slate font-medium uppercase tracking-wider">Dataset</span>
          <Select defaultValue="inspection">
            <SelectTrigger className="h-6 p-0 border-0 shadow-none bg-transparent hover:bg-transparent focus:ring-0 text-[14px] font-medium text-ink w-auto gap-1 [&>svg]:opacity-50">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="inspection">Inspection Set</SelectItem>
              <SelectItem value="training">Training Set</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="flex flex-col gap-1">
          <span className="text-[10px] text-slate font-medium uppercase tracking-wider">Environment</span>
          <Select defaultValue="production">
            <SelectTrigger className="h-6 p-0 border-0 shadow-none bg-transparent hover:bg-transparent focus:ring-0 text-[14px] font-medium text-ink w-auto gap-1 [&>svg]:opacity-50">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="production">Production</SelectItem>
              <SelectItem value="staging">Staging</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

    </header>
  );
}
