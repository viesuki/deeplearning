---
title: Operations
enableToc: True
---

# Daily Use
```bash
npx quartz build --serve
npx quartz sync
```

# Initialization
1. Install node.js & npm, then check the version
~~~bash
node -v
npm -v
git --version
~~~

2. Clone the repo & install dependencies
~~~bash
git clone https://github.com/jackyzha0/quartz.git
cd quartz
npm i
npx quartz create
~~~

3. Start the server
```bash
npx quartz build --serve   # open http://localhost:8080
npx quartz build           # only build static site
```

# Deploy to Github Pages
1. Create a new repo named `yourname`
2. Add a remote
```bash
git remote -v
git remote set-url origin <your repo REMOTE-URL>
git remote add upstream https://github.com/jackyzha0/quartz.git
```

3. Push to github
```bash
npx quartz sync --no-pull   # first time
npx quartz sync             # update
```

4. Setup Github Pages and connect to your domain

    If you have a custom domain, you can set it up in `quartz.config.ts`->`baseUrl`. For me, I did so.

5. Update Quartz version
```bash
git remote add upstream https://github.com/jackyzha0/quartz.git # first time
git fetch upstream && git merge upstream/v4  # then you go
```

