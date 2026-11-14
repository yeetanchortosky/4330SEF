import { createFileRoute } from "@tanstack/react-router";
import { Studio } from "@/components/studio/app";

export const Route = createFileRoute("/")({ component: Studio });
