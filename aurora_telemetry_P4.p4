#include <core.p4>
#include <v1model.p4>
header ethernet_t { bit<48> dstAddr; bit<48> srcAddr; bit<16> etherType; }
header ipv4_t { bit<4> version; bit<4> ihl; bit<8> diffserv; bit<16> totalLen;
 bit<16> identification; bit<3> flags; bit<13> fragOffset; bit<8> ttl;
 bit<8> protocol; bit<16> hdrChecksum; bit<32> srcAddr; bit<32> dstAddr; }
struct headers_t { ethernet_t ethernet; ipv4_t ipv4; }
struct metadata_t { }
parser ParserImpl(packet_in packet, out headers_t hdr, inout metadata_t meta,
 inout standard_metadata_t standard_metadata) {
 state start { packet.extract(hdr.ethernet); transition select(hdr.ethernet.etherType) { 0x0800: parse_ipv4; default: accept; } }
 state parse_ipv4 { packet.extract(hdr.ipv4); transition accept; }
}
control VerifyChecksumImpl(inout headers_t hdr, inout metadata_t meta) { apply {} }
control IngressImpl(inout headers_t hdr, inout metadata_t meta,
 inout standard_metadata_t standard_metadata) {
 counter(1024, CounterType.packets_and_bytes) packets_per_hash;
 action drop() { mark_to_drop(standard_metadata); }
 action ipv4_forward(bit<48> dst_mac, bit<9> port) {
  hdr.ethernet.srcAddr=hdr.ethernet.dstAddr;
  hdr.ethernet.dstAddr=dst_mac;
  standard_metadata.egress_spec=port;
  hdr.ipv4.ttl=hdr.ipv4.ttl-1;
 }
 table ipv4_src_policy { key={hdr.ipv4.srcAddr: exact;} actions={drop;NoAction;} size=1024; default_action=NoAction(); }
 table ipv4_lpm { key={hdr.ipv4.dstAddr: lpm;} actions={ipv4_forward;drop;NoAction;} size=1024; default_action=drop(); }
 apply {
  if(hdr.ipv4.isValid()) {
   bit<32> idx=hdr.ipv4.srcAddr & 1023;
   packets_per_hash.count(idx);
   if(hdr.ipv4.ttl<=1) { drop(); }
   else { if(!ipv4_src_policy.apply().hit) { ipv4_lpm.apply(); } }
  }
 }
}
control EgressImpl(inout headers_t hdr, inout metadata_t meta,
 inout standard_metadata_t standard_metadata) { apply {} }
control ComputeChecksumImpl(inout headers_t hdr, inout metadata_t meta) {
 apply { update_checksum(hdr.ipv4.isValid(),
 {hdr.ipv4.version,hdr.ipv4.ihl,hdr.ipv4.diffserv,hdr.ipv4.totalLen,hdr.ipv4.identification,
 hdr.ipv4.flags,hdr.ipv4.fragOffset,hdr.ipv4.ttl,hdr.ipv4.protocol,hdr.ipv4.srcAddr,hdr.ipv4.dstAddr},
 hdr.ipv4.hdrChecksum,HashAlgorithm.csum16); }
}
control DeparserImpl(packet_out packet, in headers_t hdr) { apply {packet.emit(hdr.ethernet);packet.emit(hdr.ipv4);} }
V1Switch(ParserImpl(),VerifyChecksumImpl(),IngressImpl(),EgressImpl(),ComputeChecksumImpl(),DeparserImpl()) main;
