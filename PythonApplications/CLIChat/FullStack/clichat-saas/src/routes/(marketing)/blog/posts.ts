export const blogInfo = {
  name: "CLIChat Blog",
  description: "CLIChat blog",
}

export type BlogPost = {
  link: string
  date: string // date is a string 'YYYY-MM-DD'
  title: string
  description: string
  parsedDate?: Date // Optional because it's added dynamically
}

// Update this list with the actual blog post list
// Create a page in the "(posts)" directory for each entry
const blogPosts: BlogPost[] = [
  {
    title: "Why CLIChat? Blog post",
    description: "Why I wanted to make CLIChat",
    link: "/blog/why_clichat_blog_post",
    date: "2025-01-02",
  },
  {
    title: "Quick \"Getting Started\"",
    description: "Get started quickly with CLIChat",
    link: "/blog/quick_getting_started",
    date: "2025-01-19",
  }
]

// Parse post dates from strings to Date objects
for (const post of blogPosts) {
  if (!post.parsedDate) {
    const dateParts = post.date.split("-")
    post.parsedDate = new Date(
      parseInt(dateParts[0]),
      parseInt(dateParts[1]) - 1,
      parseInt(dateParts[2]),
    ) // Note: months are 0-based
  }
}

export const sortedBlogPosts = blogPosts.sort(
  (a: BlogPost, b: BlogPost) =>
    (b.parsedDate?.getTime() ?? 0) - (a.parsedDate?.getTime() ?? 0),
)
