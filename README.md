# Celebration Event & Venue Website

Premium static website package for Celebration Event & Venue at 5335 Central Avenue Pike, Knoxville, Tennessee 37912.

## Overview

This is a deploy-ready HTML, CSS, and vanilla JavaScript website for a luxury indoor event venue. It includes a multi-page architecture, responsive layouts, gallery filtering, a lightbox, inquiry form markup, SEO foundations, robots.txt, sitemap.xml, favicon placeholder, and local image assets.

## File Structure

```text
.
|-- index.html
|-- about.html
|-- events.html
|-- gallery.html
|-- pricing.html
|-- faq.html
|-- contact.html
|-- css/
|   `-- site.css
|-- js/
|   |-- site.js
|   `-- gallery.js
|-- assets/
|   |-- celebration-logo.png
|   |-- venue-main.jpg
|   |-- venue-dramatic.png
|   `-- favicon.svg
|-- robots.txt
|-- sitemap.xml
`-- README.md
```

## Deploy on GitHub

1. Create a new GitHub repository.
2. Upload all files and folders from this project root.
3. Commit the files to the main branch.
4. If using GitHub Pages, enable Pages from repository settings and select the main branch root.

## Deploy on Vercel

1. Push this folder to GitHub.
2. In Vercel, choose New Project.
3. Import the repository.
4. Use the default static project settings.
5. Deploy.

## Deploy on Netlify

1. Push this folder to GitHub.
2. In Netlify, choose Add new site.
3. Import from GitHub.
4. Leave build command blank.
5. Set publish directory to the project root.
6. Deploy.

## Image Updates

Replace or add venue images in the `assets/` folder.

Primary image references:

- `assets/venue-main.jpg`
- `assets/venue-dramatic.png`
- `assets/celebration-logo.png`

Update image paths in the HTML files if you add more photography.

## Contact Info

The current phone number is `865.900.8800`.

To update it, search all HTML files for:

```text
865.900.8800
8659008800
```

## Inquiry Form

The form is located in `contact.html`.

The JavaScript in `js/site.js` currently opens the visitor's email app with the inquiry details. To connect a backend later, replace the submit handler with your form provider, CRM endpoint, Netlify Forms setup, Formspree endpoint, or custom API.

## SEO Settings

Update these before launch:

- Replace `https://celebrationeventvenue.com/` in HTML Open Graph tags, `robots.txt`, and `sitemap.xml` if the final production domain is different.
- Add the final venue street address if available.
- Add final Google Business Profile and social links if available.
- Replace sample testimonials with real testimonials when collected.

## Notes

The site is static, lightweight, dependency-free, and ready for GitHub, Vercel, Netlify, or any standard static hosting provider.
