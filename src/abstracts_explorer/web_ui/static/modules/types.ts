/**
 * Shared TypeScript types for the web UI.
 */

/** A paper as returned by the backend API. */
export interface Paper {
    uid: string;
    title?: string;
    authors?: string[];
    abstract?: string;
    conference?: string;
    session?: string;
    poster_position?: string;
    poster_image_url?: string;
    paper_pdf_url?: string;
    url?: string;
    year?: number | string;
    distance?: number;
    searchTerm?: string;
    priority?: number;
    [key: string]: unknown;
}

/** A stored rating for a paper. */
export interface PaperPriority {
    priority: number;
    searchTerm: string;
}

/** A single point in a cluster visualization. */
export interface ClusterPoint {
    x: number;
    y: number;
    title?: string;
    cluster?: number | string;
    cluster_id?: number | string;
}

/** A visualization descriptor returned by an MCP tool. */
export interface Visualization {
    type?: string;
    topics?: string[];
    conference_data?: Record<string, Record<string, { year_relative?: Record<string, number> }>>;
    points?: ClusterPoint[];
    statistics?: {
        total_papers?: number;
        n_clusters?: number;
        [key: string]: unknown;
    };
    [key: string]: unknown;
}

/** Metadata attached to a chat response. */
export interface ChatMetadata {
    rewritten_query?: string;
    retrieved_new_papers?: boolean;
    [key: string]: unknown;
}
