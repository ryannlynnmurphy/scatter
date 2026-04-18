# Home Data Center: k3s + Ollama on 4× Pi 5

Turn four Pi 5s into a distributed inference cluster that the Scatter router can talk to. Uses **k3s** (lightweight Kubernetes) because it's what people actually run on edge/home hardware — not full Kubernetes, which would melt a Pi.

Estimated total time (all stages, first run): **3–4 hours.**

## Prerequisites

Physical:
- 4× Raspberry Pi 5 (8GB recommended), each with power + microSD
- Gigabit ethernet switch + 4 ethernet cables
- Uplink cable from switch to your home router
- Laptop on same network
- USB SD card reader (for the 4th Pi that still needs flashing)

Software on laptop:
- SSH keys (if you push to GitHub via SSH, you have them)
- `kubectl` (we'll install below)

## Stage 1 — Flash + network the 4th Pi

On your laptop:

```bash
# Install Raspberry Pi Imager if you don't have it
sudo apt install rpi-imager  # or: flatpak install flathub org.raspberrypi.rpi-imager
```

In Imager:
- OS: "Raspberry Pi OS Lite (64-bit)" — no desktop, leaner, faster boot
- Advanced settings (gear icon):
  - Hostname: `pi4`
  - Enable SSH: yes, use public-key auth, paste `~/.ssh/id_ed25519.pub` (or `id_rsa.pub`)
  - Set username: your username, set password
- Write. Takes ~5 min.

Plug the Pi into the switch via ethernet. Power on.

## Stage 2 — Get SSH working on all 4 Pis

The 3 already-flashed Pis are on WiFi. Move them to ethernet too:

```bash
# On each Pi via direct monitor+keyboard (one-time):
sudo raspi-config
# → System Options → Hostname → set to pi1, pi2, pi3 respectively
# → Interface Options → SSH → enable
# reboot
```

Plug all 4 into the switch.

On your laptop, find their IPs. Either check your home router's admin page (DHCP clients list) or:

```bash
sudo apt install arp-scan
sudo arp-scan --localnet | grep -i raspberry
```

Note the 4 IPs. Copy your SSH key to each:

```bash
for ip in <pi1-ip> <pi2-ip> <pi3-ip> <pi4-ip>; do
    ssh-copy-id $USER@$ip
done
```

Verify you can SSH passwordless:

```bash
for ip in <pi1-ip> <pi2-ip> <pi3-ip> <pi4-ip>; do
    ssh $USER@$ip "hostname && uptime"
done
```

**Reserve IPs on your home router** (admin page → DHCP reservations, map each Pi's MAC to a permanent IP). This is critical — without it, IPs can change on reboot and break k3s.

## Stage 3 — Install k3s server on pi1

```bash
ssh $USER@<pi1-ip>

# Install k3s as server. Writes kubeconfig to /etc/rancher/k3s/k3s.yaml.
curl -sfL https://get.k3s.io | sh -

# Verify
sudo kubectl get nodes
# should show pi1 as a single Ready node

# Grab the token (needed for other Pis to join)
sudo cat /var/lib/rancher/k3s/server/node-token
# copy this string somewhere — you'll need it in Stage 4

exit
```

## Stage 4 — Join pi2, pi3, pi4 as agents

From your laptop, SSH to each of pi2/pi3/pi4 and run:

```bash
curl -sfL https://get.k3s.io | K3S_URL=https://<pi1-ip>:6443 K3S_TOKEN=<token-from-stage-3> sh -
```

After all three join, verify from pi1:

```bash
ssh $USER@<pi1-ip> "sudo kubectl get nodes"
# Should show 4 nodes: pi1 (control-plane,master), pi2/3/4 (<none>)
# All Ready.
```

## Stage 5 — Control k3s from your laptop

Copy the kubeconfig down and rewrite the server address so it points at pi1 (not 127.0.0.1):

```bash
mkdir -p ~/.kube
scp $USER@<pi1-ip>:/etc/rancher/k3s/k3s.yaml ~/.kube/config
sed -i "s/127.0.0.1/<pi1-ip>/" ~/.kube/config
chmod 600 ~/.kube/config

# Install kubectl on laptop
sudo apt install kubectl  # or snap install kubectl --classic

kubectl get nodes
# Same output as running it on pi1
```

## Stage 6 — Deploy Ollama as a DaemonSet

One pod per node. Each pod pulls qwen2.5-coder:1.5b into its own volume. Service exposes them together.

Save this on your laptop as `~/scatter/docs/ollama-k8s.yaml`:

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: ollama
  namespace: default
spec:
  selector:
    matchLabels:
      app: ollama
  template:
    metadata:
      labels:
        app: ollama
    spec:
      containers:
      - name: ollama
        image: ollama/ollama:latest
        ports:
        - containerPort: 11434
          name: http
        volumeMounts:
        - name: models
          mountPath: /root/.ollama
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "6Gi"
      volumes:
      - name: models
        hostPath:
          path: /var/lib/ollama
          type: DirectoryOrCreate
---
apiVersion: v1
kind: Service
metadata:
  name: ollama
  namespace: default
spec:
  type: NodePort
  selector:
    app: ollama
  ports:
  - port: 11434
    targetPort: 11434
    nodePort: 31134
```

Apply:

```bash
kubectl apply -f ~/scatter/docs/ollama-k8s.yaml
kubectl get pods -l app=ollama
# Wait for all 4 to go Running. First pull of image takes a few min.
```

Pull the model on each node (once):

```bash
for ip in <pi1-ip> <pi2-ip> <pi3-ip> <pi4-ip>; do
    ssh $USER@$ip "sudo docker exec \$(sudo docker ps -q --filter name=ollama) ollama pull qwen2.5-coder:1.5b" &
done
wait
```

(Or: use a Kubernetes Job to pull on each node. DaemonSet pattern above is simpler but less declarative.)

## Stage 7 — Update the router to talk to the cluster

Edit `~/scatter/server.py`:

```python
# Change this line:
OLLAMA = "http://127.0.0.1:11434/api/generate"

# To something like (pick one node — k8s load-balances internally):
OLLAMA = "http://<pi1-ip>:31134/api/generate"

# Or better: round-robin across all four (real parallelism for julienne later):
import random
OLLAMA_NODES = [
    "http://<pi1-ip>:31134/api/generate",
    "http://<pi2-ip>:31134/api/generate",
    "http://<pi3-ip>:31134/api/generate",
    "http://<pi4-ip>:31134/api/generate",
]
OLLAMA = random.choice(OLLAMA_NODES)  # or rotate per call
```

Restart uvicorn. Test from the web UI — `local:qwen` should now resolve against the cluster instead of the laptop.

## Verification

- `kubectl get nodes` → 4 Ready
- `kubectl get pods -l app=ollama` → 4 Running
- `curl http://<any-pi-ip>:31134/api/tags` → returns `qwen2.5-coder:1.5b` in the model list
- Scatter UI chat with `local only` checked → response comes back, watts logged

## What this unlocks

- **Julienne becomes real.** The parallelism that was theater on one laptop is real across 4 nodes.
- **Resilience.** If any Pi crashes, k3s restarts the pod. If a node dies, the other 3 still serve.
- **Declarative upgrades.** Change the image tag in the YAML, `kubectl apply`, done.
- **Transferable skill.** `kubectl` is what Alex uses for fintech at scale. Same commands, different cluster.

## Known limitations (tonight's scope)

- No GPU (Pi 5 has none) — inference still CPU-only, just distributed.
- 4× Pi 5 with 1.5b model is roughly 4× the throughput of one Pi, not 4× the intelligence.
- Model pull per-node is manual on first run — Stage 7 could be smoother with a proper Job resource.
- Ethernet via consumer switch; no redundant networking. If the switch dies, the whole cluster is offline.

## If it breaks

- `kubectl describe pod <pod-name>` — what went wrong
- `kubectl logs <pod-name>` — what the pod said
- `sudo journalctl -u k3s -n 50` (on server) or `-u k3s-agent` (on agents)
- Reset worst case: `/usr/local/bin/k3s-uninstall.sh` on server, `/usr/local/bin/k3s-agent-uninstall.sh` on agents, start over.

## Next after this ships

- **Julienne for real**: edit the router to chunk long inputs and fan out to the 4 nodes in parallel.
- **Model auto-pull**: convert manual `ollama pull` into an init container or Job.
- **Real wattmeter**: Kill A Watt between the switch and the wall → real IPW numbers replace estimates.
- **Monitoring**: `kube-prometheus-stack` chart for Grafana dashboards of node CPU/RAM/throughput. Skip for v0.
