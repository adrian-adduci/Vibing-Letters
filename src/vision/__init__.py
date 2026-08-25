"""Pixels to geometry.

The codec works on radius profiles; images are what actually get shipped. This
package is the bridge, and it is deliberately the only place that knows about
both. The browser decoder reimplements exactly this contract.
"""
